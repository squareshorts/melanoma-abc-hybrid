from pathlib import Path
import json, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
R=Path('/mnt/data/melanoma_reanalysis_bundle_extracted'); O=Path('/mnt/data/melanoma_reanalysis_results'); seed=20261003
ham=pd.read_csv(R/'external_metadata/HAM10000_metadata.csv');ham['y_true']=(ham.dx=='mel').astype(int)
with open(R/'data/splits/ham_lesion_split.json') as f:sp=json.load(f)
E=np.load(R/'data/derived/embeddings/ham_effb0.npy',mmap_mode='r');ids=pd.read_csv(R/'data/derived/embeddings/ham_effb0_ids.csv').image_id.astype(str).tolist();mp={x:i for i,x in enumerate(ids)}
tr=ham[ham.lesion_id.isin(set(sp['train']))][['image_id','y_true']].drop_duplicates();tr=tr[tr.image_id.isin(mp)].copy();X=np.asarray(E[[mp[i] for i in tr.image_id]])
model=xgb.XGBClassifier(n_estimators=600,max_depth=4,learning_rate=.03,subsample=.9,colsample_bytree=.9,reg_lambda=1.0,eval_metric='logloss',n_jobs=4,random_state=0)
model.fit(X,tr.y_true.to_numpy())
Eb=np.load(R/'data/derived/embeddings/bcn20000_effb0.npy',mmap_mode='r');bi=pd.read_csv(R/'data/derived/embeddings/bcn20000_effb0_ids.csv').image_id.astype(str).tolist();bm={x:i for i,x in enumerate(bi)}
bcn=pd.read_csv(O/'bcn_predictions_metadata_clustered.csv'); d=bcn[bcn.image_id.isin(bm)][['image_id','lesion_id','y_true','Hybrid']].copy();Xb=np.asarray(Eb[[bm[i] for i in d.image_id]]);d['Embeddings-only']=model.predict_proba(Xb)[:,1]
a=roc_auc_score(d.y_true,d.Hybrid); b=roc_auc_score(d.y_true,d['Embeddings-only'])
# paired lesion bootstrap 500
les=d.lesion_id.astype(str).to_numpy();uniq,codes=np.unique(les,return_inverse=True);k=len(uniq);rng=np.random.default_rng(seed);delta=[]
for _ in range(500):
 cnt=np.bincount(rng.integers(0,k,size=k),minlength=k).astype(float);w=cnt[codes]
 delta.append(roc_auc_score(d.y_true,d.Hybrid,sample_weight=w)-roc_auc_score(d.y_true,d['Embeddings-only'],sample_weight=w))
lo,hi=np.quantile(delta,[.025,.975])
pd.DataFrame([{'model':'Embeddings-only','AUC':b},{'model':'Hybrid','AUC':a}]).to_csv(O/'table_external_embeddings_only_ablation.csv',index=False)
pd.DataFrame([{'comparison':'Hybrid minus Embeddings-only','delta_AUC':a-b,'ci_low':lo,'ci_high':hi,'inference':'paired lesion-clustered bootstrap','n_boot':500}]).to_csv(O/'table_external_embeddings_only_delta.csv',index=False)
print('AUC embeddings-only',b,'hybrid',a,'delta',a-b,'CI',lo,hi)
