#!/usr/bin/env python3
"""Monte Carlo validation of lesion-cluster and source-threshold uncertainty.

Fixed population for continuous score differences; 95% percentile intervals.
The source-reestimated interval targets a population source policy; the
fixed-cutoff interval targets the realized cutoff's population sensitivity.
Run: python synthetic_validation.py --out simulated_validation.json
"""
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

def auc(y,score):
    pos = (y==1)
    npos=pos.sum(); nneg=len(y)-npos
    return float((rankdata(score,method='average')[pos].sum() - npos*(npos+1)/2)/(npos*nneg))

def bootstrap_auc(rng,y,s,groups,B):
    n=len(groups)
    cl=np.empty(B)
    for b in range(B):
        idx=rng.integers(0,n,n)
        flat=groups[idx].ravel()
        cl[b]=auc(y[flat],s[flat])
    return np.percentile(cl,[2.5,97.5])

def naive_bootstrap_auc(rng,y,s,B):
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    out=np.empty(B)
    for b in range(B):
        ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        out[b]=auc(y[ix],s[ix])
    return np.percentile(out,[2.5,97.5])

def cutoff_for_specificity(sneg, target=.90):
    # For an empirical source sample, choose the lowest cutoff that retains
    # at least target specificity under positive-if-score>=cutoff.
    scores=np.unique(sneg)
    for t in scores:
        if np.mean(sneg<t)>=target:
            return float(t)
    return float('inf')

def interval_rates(rng,src_neg,src_pos,ex_neg,ex_pos,B,reselect):
    if not reselect:
        t=cutoff_for_specificity(src_neg)
    out=np.empty(B)
    for b in range(B):
        en=rng.choice(ex_neg,len(ex_neg),replace=True) # reproducibility of external sample
        ep=rng.choice(ex_pos,len(ex_pos),replace=True)
        if reselect:
            sn=rng.choice(src_neg,len(src_neg),replace=True)
            # positive labels affect sensitivity but specific rule only uses negative class
            t=cutoff_for_specificity(sn)
        out[b]=np.mean(ep>=t)
    return np.percentile(out,[2.5,97.5])

def run(seed=20261110, simulations=120,bootstraps=220):
    rng=np.random.default_rng(seed)
    n_pos,n_neg,m=35,90,4
    mu=1.0;sb=1.0;sw=.7
    true_auc=float(ndtr(mu/math.sqrt(2*(sb**2+sw**2))))
    n_p=35;n_n=90
    true_t=float(ndtri(.90)); pop_tpr=float(ndtr(mu-true_t))
    auc_cover=np.zeros(2);auc_width=np.zeros(2)
    rate_cover=np.zeros(2);rate_width=np.zeros(2)
    for _ in range(simulations):
        yy=np.r_[np.ones(n_pos,dtype=int), np.zeros(n_neg,dtype=int)]
        z=np.r_[rng.normal(mu,sb,n_pos),rng.normal(0,sb,n_neg)]
        xx=np.repeat(z,m)+rng.normal(0,sw,len(z)*m)
        y=np.repeat(yy,m)
        groups=np.arange(len(z)*m).reshape(-1,m)
        # Cluster-stratified lesion resampling avoids rare all-one-class draws.
        cls=np.r_[np.arange(n_pos),np.arange(n_pos,len(z))]
        ac=np.empty(bootstraps)
        for b in range(bootstraps):
            sampled=np.r_[rng.choice(cls[:n_pos],n_pos,replace=True),rng.choice(cls[n_pos:],n_neg,replace=True)]
            take=groups[sampled].ravel()
            ac[b]=auc(y[take],xx[take])
        ci=np.percentile(ac,[2.5,97.5]); ci2=naive_bootstrap_auc(rng,y,xx,bootstraps)
        for i,ci0 in enumerate((ci,ci2)):
            auc_cover[i]+=(ci0[0]<=true_auc<=ci0[1]);auc_width[i]+=(ci0[1]-ci0[0])
        srcneg=rng.normal(0,1,100);srcpos=rng.normal(mu,1,45)
        extneg=rng.normal(0,1,100); extpos=rng.normal(mu,1,45)
        t=cutoff_for_specificity(srcneg)
        target_fixed=float(ndtr(mu-t))
        cint=interval_rates(rng,srcneg,srcpos,extneg,extpos,bootstraps,False)
        cire=interval_rates(rng,srcneg,srcpos,extneg,extpos,bootstraps,True)
        for i,(cc,target) in enumerate(((cint,target_fixed),(cire,pop_tpr))):
            rate_cover[i]+=(cc[0]<=target<=cc[1]); rate_width[i]+=(cc[1]-cc[0])
    return {
      'seed':seed,'simulations':simulations,'bootstraps_per_simulation':bootstraps,
      'confidence_level':.95,
      'data_generating_process':{'n_positive_lesions':n_pos,'n_negative_lesions':n_neg,'views_per_lesion':m,'positive_shift':mu,'between_lesion_sd':sb,'within_lesion_sd':sw},
      'population_image_auc':true_auc,
      'image_auc_clustered_coverage':auc_cover[0]/simulations,
      'image_auc_naive_independent_coverage':auc_cover[1]/simulations,
      'image_auc_clustered_mean_width':auc_width[0]/simulations,
      'image_auc_naive_mean_width':auc_width[1]/simulations,
      'source_specificity_target':.9,'population_target_sensitivity':pop_tpr,
      'fixed_cutoff_CI_conditional_coverage':rate_cover[0]/simulations,
      'source_reestimated_CI_population_policy_coverage':rate_cover[1]/simulations,
      'fixed_cutoff_CI_mean_width':rate_width[0]/simulations,
      'source_reestimated_CI_mean_width':rate_width[1]/simulations,
      'interpretation':'Monte Carlo screening of interval behavior under specified normal random-effects data-generating processes, not a proof of nominal coverage in other conditions.'
    }

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--out',default='simulation_results.json');pa.add_argument('--simulations',type=int,default=120);pa.add_argument('--bootstraps',type=int,default=220);pa.add_argument('--seed',type=int,default=20261110)
    a=pa.parse_args();d=run(a.seed,a.simulations,a.bootstraps);Path(a.out).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))