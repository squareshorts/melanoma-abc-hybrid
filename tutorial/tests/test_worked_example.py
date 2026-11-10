"""Checks for score-level protocol and included reference outputs."""
import os, sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import worked_example as w

class ProtocolTests(unittest.TestCase):
    def test_tie_aware_cutoff(self):
        y=np.array([1,1,0,0,0,0]); p=np.array([.9,.7,.7,.6,.4,.4])
        t=w.cutoff(y,p,'sensitivity',1.0)
        self.assertEqual(t,.7)
        self.assertEqual(w.operating(y,p,t),(1.0,.75))

    def test_no_external_cutoff_leakage(self):
        y=np.array([1,1,0,0]);src=np.array([.8,.6,.5,.2]);ext=np.array([.4,.3,.39,.1])
        t=w.cutoff(y,src,'sensitivity',1)
        self.assertEqual(w.operating(y,ext,t)[0],0)
        self.assertEqual(w.operating(y,ext,w.cutoff(y,ext,'sensitivity',1))[0],1)

    def test_one_lesion_one_probability(self):
        x=pd.DataFrame({'lesion_id':['a','a','b','c'],'image_id':['1','2','3','4'],
             'y_true':[1,1,0,0],'Handcrafted':[.8,.6,.1,.2],
             'Deep':[.7,.5,.2,.3],'Hybrid':[.9,.7,.1,.1]})
        f=w.lesions(x)
        self.assertEqual(len(f),3)
        self.assertEqual(int(f.n_images.sum()),4)
        self.assertAlmostEqual(float(f.loc[f.lesion_id.eq('a'),'Deep'].iloc[0]),.6)

    def test_reject_conflicting_labels(self):
        import tempfile
        x=pd.DataFrame({'lesion_id':['a','a','b','c'],'image_id':['1','2','3','4'],
             'y_true':[1,0,0,1],'Handcrafted':[.8,.6,.1,.2],
             'Deep':[.7,.5,.2,.3],'Hybrid':[.9,.7,.1,.1]})
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'scores.csv';x.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'contradictory'):w.read_scores(p)

    def test_group_index_completeness(self):
        x=pd.DataFrame({'lesion_id':['a','a','b','c','c','c']})
        groups=w.cluster_indices(x)
        self.assertEqual(sorted(map(len,groups)),[1,2,3])
        self.assertEqual(sorted(i for group in groups for i in group),list(range(6)))

    def test_reference_outputs(self):
        root=Path(os.environ.get('TUTORIAL_RESULT_DIR', str(Path(__file__).resolve().parents[1]/'results')))
        df=pd.read_csv(root/'threshold_transport.csv')
        x=df[(df.unit=='lesion')&(df.source_target=='sensitivity')].set_index('model')
        self.assertGreater(x.loc['Handcrafted','external_sensitivity'],x.loc['Deep','external_sensitivity'])
        self.assertGreater(x.loc['Deep','external_sensitivity'],x.loc['Hybrid','external_sensitivity'])
        self.assertGreater(x.loc['Hybrid','external_auc'],x.loc['Deep','external_auc'])
        self.assertGreater(x.loc['Deep','external_auc'],x.loc['Handcrafted','external_auc'])
        ci=pd.read_csv(root/'bootstrap_comparison.csv')
        self.assertTrue(np.allclose(ci['estimate'].values, np.array([0.7126250829075569, 0.7126250829075569, 0.760814299086658, 0.760814299086658]), atol=1e-12))
        for model in ('Deep','Hybrid'):
            v=ci[ci.model.eq(model)].set_index('method')
            self.assertGreater(v.loc['clustered_lesions','width'],v.loc['naive_images','width'])

if __name__=='__main__':unittest.main()