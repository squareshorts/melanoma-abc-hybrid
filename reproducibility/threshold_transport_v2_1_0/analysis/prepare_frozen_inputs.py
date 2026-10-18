#!/usr/bin/env python3
"""Align the three archived fitted-model scores; fail rather than mix splits.

This script never fits a model or changes a probability. It keeps the archived
partitions as a reference, writes an exclusion-only sensitivity set, and writes
the primary evaluation set with known training-linked held-out groups removed
and retained published same-lesion components merged. The fitted CNN remains
conditional on its historical training and checkpoint selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MODELS = ('Handcrafted', 'Deep', 'Hybrid')
KEYS = ['image_id', 'lesion_id', 'y_true']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def prepare(bundle, audit, reanalysis, exclusions, out):
    sources=[]
    def read(path):
        path=Path(path)
        sources.append({'path':str(path), 'sha256':sha(path)})
        return pd.read_csv(path)
    splitpath=bundle/'data/splits/ham_lesion_split.json'
    split=json.loads(splitpath.read_text())
    sources.append({'path':str(splitpath),'sha256':sha(splitpath)})
    meta=read(bundle/'external_metadata/HAM10000_metadata.csv')
    assert not meta.image_id.duplicated().any()
    meta['y_true']=(meta.dx=='mel').astype(int)
    val=read(audit/'results/final/source_validation_predictions_all_models.csv')[KEYS+list(MODELS)]
    test=None
    for label,folder,suffix in [('Handcrafted','handcrafted','handcrafted'),('Deep','deep_baseline','deep'),('Hybrid','hybrid','hybrid')]:
        df=read(bundle/f'results/runs/{folder}/ham_test_predictions_{suffix}.csv')
        assert not df.image_id.duplicated().any()
        df=df[KEYS+['y_prob']].rename(columns={'y_prob':label})
        if test is None:
            test=df
        else:
            assert set(test.image_id)==set(df.image_id), 'Test model image IDs differ'
            test=test.merge(df, on=KEYS, validate='one_to_one', how='inner')
            assert len(test)==1510, 'Conflicting test labels/lesion IDs'
    bcn=read(reanalysis/'bcn_predictions_metadata_clustered.csv').rename(columns={'ABC-only':'Handcrafted'})[KEYS+list(MODELS)]
    assert (len(val),len(test),len(bcn))==(1505,1510,12413)
    expected={'ham_validation':{'Handcrafted':0.8297466808562606,'Deep':0.9225104241176573,'Hybrid':0.9244709433172247},
              'ham_test':{'Handcrafted':0.805761522375948,'Deep':0.8873110071740361,'Hybrid':0.884751717711264},
              'bcn_external':{'Handcrafted':0.6916010487631957,'Deep':0.7126250829075569,'Hybrid':0.760814299086658}}
    frames={'ham_validation':val,'ham_test':test,'bcn_external':bcn}
    summary={}
    for name,df in frames.items():
        assert not df.image_id.duplicated().any()
        assert df.groupby('lesion_id').y_true.nunique().max()==1
        assert np.isfinite(df[list(MODELS)].to_numpy()).all()
        assert df[list(MODELS)].ge(0).all().all() and df[list(MODELS)].le(1).all().all()
        if name!='bcn_external':
            partition='val' if name=='ham_validation' else 'test'
            ref=meta.loc[meta.lesion_id.isin(split[partition]),KEYS]
            assert set(df.image_id)==set(ref.image_id), f'{name}: not the archived split'
            assert len(df.merge(ref,on=KEYS,validate='one_to_one'))==len(df)
        aucs={m:roc_auc_score(df.y_true,df[m]) for m in MODELS}
        for m,value in aucs.items():
            assert abs(value-expected[name][m])<1e-12, (name,m,value)
        summary[name]={'images':len(df),'lesions':df.lesion_id.nunique(),'melanoma_images':int(df.y_true.sum()),'melanoma_lesions':int(df.groupby('lesion_id').y_true.first().sum()),'auc':aucs}
    # Check public labels and the exact evaluation subset independently of score files.
    bmeta=read(bundle/'external_metadata/BCN20000_metadata.csv').rename(columns={'isic_id':'image_id'})
    ref=bmeta[['image_id','lesion_id']]
    assert len(bcn.merge(ref,on=['image_id','lesion_id'],validate='one_to_one'))==len(bcn)
    for name,df in frames.items():
        directory=out/'original'
        directory.mkdir(parents=True,exist_ok=True)
        df.sort_values('image_id').to_csv(directory/f'{name}.csv',index=False,float_format='%.17g')
    excl=read(exclusions)
    components=read(Path(exclusions).parent/'ham_known_duplicate_components.csv')
    identity_map={old:row.known_component_id for row in components.itertuples() for old in row.lesion_ids.split(';')}
    sensitivity={}
    corrected_summary={}
    for name,df in frames.items():
        partition={'ham_validation':'val','ham_test':'test','bcn_external':'external'}[name]
        # Audit uses the full spelling validation; accept only these documented aliases.
        names={'val':['val','validation'],'test':['test'],'external':[]}[partition]
        ids=set(excl.loc[excl.partition.isin(names),'image_id'])
        removed=df.loc[df.image_id.isin(ids)]
        assert len(removed)==(0 if name=='bcn_external' else 6)
        assert not removed.y_true.any(), 'Audit expected benign-only duplicates'
        retained=df.loc[~df.image_id.isin(ids)]
        assert not set(retained.lesion_id)&set(removed.lesion_id), 'Remove whole affected held-out groups'
        sensitivity[name]={'excluded_images':len(removed),'excluded_lesions':removed.lesion_id.nunique(),'remaining_images':len(retained),'remaining_lesions':retained.lesion_id.nunique()}
        directory=out/'duplicate_sensitivity'
        directory.mkdir(parents=True,exist_ok=True)
        retained.sort_values('image_id').to_csv(directory/f'{name}.csv',index=False,float_format='%.17g')
        corrected=retained.copy()
        corrected['original_lesion_id']=corrected.lesion_id
        if name!='bcn_external':
            corrected['lesion_id']=corrected.lesion_id.map(lambda x:identity_map.get(x,x))
        assert corrected.groupby('lesion_id').y_true.nunique().max()==1
        expected_groups={'ham_validation':1114,'ham_test':1116,'bcn_external':3576}[name]
        assert corrected.lesion_id.nunique()==expected_groups
        corrected_summary[name]={'images':len(corrected),'lesions':corrected.lesion_id.nunique(),'melanoma_images':int(corrected.y_true.sum()),'melanoma_lesions':int(corrected.groupby('lesion_id').y_true.first().sum()),'images_with_corrected_lesion_id':int((corrected.lesion_id!=corrected.original_lesion_id).sum())}
        directory=out/'known_identity_corrected'
        directory.mkdir(parents=True,exist_ok=True)
        corrected.sort_values('image_id').to_csv(directory/f'{name}.csv',index=False,float_format='%.17g')
    for path in [bundle/'results/runs/deep_baseline/efficientnet_b0_best.pt',bundle/'results/runs/handcrafted/handcrafted_xgb.joblib',bundle/'results/runs/hybrid/hybrid_xgb.joblib']:
        sources.append({'path':str(path),'sha256':sha(path),'role':'historical fitted model; not loaded or modified'})
    output_files=[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*.csv'))]
    manifest={'purpose':'Exact alignment of three archived fitted-model score exports','models':list(MODELS),'source_files':sources,'output_files':output_files,'cohorts':summary,'duplicate_sensitivity':sensitivity,'known_identity_corrected':corrected_summary,'primary_analysis':'known_identity_corrected; original and exclusion-only runs retained as sensitivity analyses','excluded_legacy_inputs':['ham_test_predictions_deep_stratified_lesion.csv','ham_test_predictions_ablation.csv'],'limitations':['Removing known training-linked held-out groups and merging published retained identity components cannot undo historical CNN checkpoint selection.','No patient identifiers available; lesion separation is not patient separation.','No new predictions were generated in this alignment step.']}
    (out/'input_alignment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'cohorts':summary,'duplicate_sensitivity':sensitivity,'known_identity_corrected':corrected_summary},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--reanalysis',type=Path,required=True)
    p.add_argument('--exclusions',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    prepare(a.bundle,a.audit,a.reanalysis,a.exclusions,a.output_dir)
