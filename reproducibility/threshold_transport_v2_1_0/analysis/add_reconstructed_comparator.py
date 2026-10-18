#!/usr/bin/env python3
"""Add the single documented historical-spec head, preserving original scores."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main(root):
    names={'ham_validation':'ham_val','ham_test':'ham_test','bcn_external':'bcn_external'}
    manifest={'purpose':__doc__,'new_model':'EmbeddingsOnly_Reconstructed','reconstruction_manifest':'comparator/outputs/model_frozen_manifest.json','variants':{}}
    for variant in ['original','known_identity_corrected']:
        out=root/'inputs'/('four_model_'+variant);out.mkdir(exist_ok=True)
        manifest['variants'][variant]={}
        for cohort,short in names.items():
            p=root/'inputs'/variant/(cohort+'.csv')
            q=root/'comparator/outputs'/('predictions_'+short+'.csv')
            a=pd.read_csv(p);b=pd.read_csv(q)
            assert b.image_id.is_unique
            original=a[['image_id','original_lesion_id' if 'original_lesion_id' in a else 'lesion_id']].copy()
            original.columns=['image_id','lesion_id']
            assert len(original.merge(b[['image_id','lesion_id']],on=['image_id','lesion_id'],validate='one_to_one'))==len(a)
            d=a.merge(b[['image_id','y_true','y_prob']],on=['image_id','y_true'],how='left',validate='one_to_one').rename(columns={'y_prob':'EmbeddingsOnly_Reconstructed'})
            assert d['EmbeddingsOnly_Reconstructed'].notna().all() and len(a)==len(d)
            for col in ['Handcrafted','Deep','Hybrid']:
                assert (a[col].values==d[col].values).all()
            path=out/(cohort+'.csv');d.to_csv(path,index=False,float_format='%.17g')
            manifest['variants'][variant][cohort]={'input_sha256':digest(p),'comparator_sha256':digest(q),'output_sha256':digest(path),'rows':len(d),'lesions':d.lesion_id.nunique()}
    (root/'inputs/four_model_alignment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();main(a.root)
