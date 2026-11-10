#!/usr/bin/env python3
"""Reproducible score-level tutorial for melanoma external validation.

Reads only archived prediction CSVs from squareshorts/melanoma-abc-hybrid.
Does not fit, adapt, retrain, or download a classifier or use raw images.

The 2026-10-18 repository archive is the authoritative reference for its
2,000-replicate estimates. This teaching script computes additional diagnostic
comparisons using user-controlled bootstrap replicates and a fixed seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib.lines import Line2D

BASE = Path('reproducibility/threshold_transport_v2_1_0')
DATA = BASE / 'inputs/four_model_known_identity_corrected'
ORIGINAL = BASE / 'inputs/four_model_original'
MODELS = ('Handcrafted', 'Deep', 'Hybrid')
SEED = 20261110


def read_scores(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    req = {'image_id', 'lesion_id', 'y_true', *MODELS}
    if not req.issubset(x):
        raise ValueError(f'{path}: missing columns {req - set(x)}')
    if not x.image_id.is_unique or x[['image_id','lesion_id','y_true']].isna().any().any():
        raise ValueError(f'{path}: duplicate/missing identifiers or labels')
    if not x.y_true.isin([0, 1]).all() or x.y_true.nunique() < 2:
        raise ValueError(f'{path}: invalid/one-class labels')
    if (x.groupby('lesion_id').y_true.nunique() > 1).any():
        raise ValueError(f'{path}: contradictory lesion labels')
    if not np.isfinite(x[list(MODELS)].to_numpy(float)).all():
        raise ValueError(f'{path}: nonfinite scores')
    return x[['image_id','lesion_id','y_true',*MODELS]].copy()


def lesions(x: pd.DataFrame) -> pd.DataFrame:
    return x.groupby('lesion_id', as_index=False, sort=True).agg(
        y_true=('y_true', 'first'),
        n_images=('image_id', 'size'),
        **{name: (name, 'mean') for name in MODELS})


def cutoff(y: np.ndarray, score: np.ndarray, kind: str, target: float = .90) -> float:
    """Tied-score aware cutoff, consistent with the archive (predict >= threshold).

    If the constraint is sensitivity, maximize specificity (then sensitivity).
    If the constraint is specificity, maximize sensitivity (then specificity).
    Remaining ties are broken by the largest threshold.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if not (len(y) == len(score) and y.min() == 0 and y.max() == 1):
        raise ValueError('Need aligned, two-class labels and scores')
    order = np.argsort(-score, kind='stable')
    ys, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss)-1]
    tp = np.r_[0, np.cumsum(ys)[ends]]
    fp = np.r_[0, ends + 1 - np.cumsum(ys)[ends]]
    th = np.r_[np.inf, ss[ends]]
    npos = int(y.sum()); nneg = int(len(y)-npos)
    sensitivity = tp / npos; specificity = (nneg-fp)/nneg
    if kind == 'specificity':
        keep = specificity >= target - 1e-12
        first, second = tp, nneg-fp
    elif kind == 'sensitivity':
        keep = sensitivity >= target - 1e-12
        first, second = nneg-fp, tp
    else:
        raise ValueError(kind)
    positions = np.flatnonzero(keep)
    if not len(positions):
        raise ValueError('No eligible cutoff')
    positions = positions[first[positions] == first[positions].max()]
    positions = positions[second[positions] == second[positions].max()]
    return float(np.max(th[positions]))


def operating(y: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    pos = np.asarray(scores, dtype=float) >= threshold
    if y.min() != 0 or y.max() != 1:
        raise ValueError('Two outcome classes are required')
    return float(pos[y == 1].mean()), float((~pos[y == 0]).mean())


def cluster_indices(x: pd.DataFrame) -> list[np.ndarray]:
    return [np.asarray(ind, dtype=int) for ind in x.groupby('lesion_id', sort=True).indices.values()]


def auc_interval(x: pd.DataFrame, model: str, method: str, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    yy = x.y_true.to_numpy(int); scores = x[model].to_numpy(float)
    groups = cluster_indices(x)
    bootstrap = np.empty(reps)
    for r in range(reps):
        if method == 'naive_images':
            ind = rng.integers(len(x), size=len(x))
        elif method == 'clustered_lesions':
            sampling = rng.integers(len(groups), size=len(groups))
            ind = np.concatenate([groups[k] for k in sampling])
        else:
            raise ValueError(method)
        if np.unique(yy[ind]).size < 2:
            raise ValueError('Bootstrap sample lost an outcome class')
        bootstrap[r] = roc_auc_score(yy[ind], scores[ind])
    return np.quantile(bootstrap, [.025, .975]).tolist()


def nested_delta(src: pd.DataFrame, ext: pd.DataFrame, kind: str, reps: int,
                 reselect: bool, seed: int) -> tuple[float,float]:
    """Paired Hybrid minus Deep sensitivity, cluster-bootstrap by lesion."""
    r = np.random.default_rng(seed)
    sc = cluster_indices(src); ec = cluster_indices(ext)
    sy = src.y_true.to_numpy(int); ey = ext.y_true.to_numpy(int)
    ss = {m: src[m].to_numpy(float) for m in ('Deep','Hybrid')}
    es = {m: ext[m].to_numpy(float) for m in ('Deep','Hybrid')}
    baseline = {m: cutoff(sy, ss[m], kind, .90) for m in ss}
    values = np.empty(reps)
    for b in range(reps):
        si = np.concatenate([sc[k] for k in r.integers(len(sc), size=len(sc))]) if reselect else None
        ei = np.concatenate([ec[k] for k in r.integers(len(ec), size=len(ec))])
        if reselect and np.unique(sy[si]).size < 2:
            raise ValueError('Source bootstrap one-class sample')
        if np.unique(ey[ei]).size < 2:
            raise ValueError('External bootstrap one-class sample')
        chosen = {m: cutoff(sy[si], ss[m][si], kind, .90) for m in ss} if reselect else baseline
        values[b] = (operating(ey[ei],es['Hybrid'][ei],chosen['Hybrid'])[0]
                     - operating(ey[ei],es['Deep'][ei],chosen['Deep'])[0])
    return tuple(float(v) for v in np.quantile(values, [.025,.975]))


def points(src: pd.DataFrame, ext: pd.DataFrame, unit: str) -> list[dict]:
    rows = []
    for name in MODELS:
        for policy in ('sensitivity','specificity'):
            t = cutoff(src.y_true.to_numpy(int), src[name].to_numpy(float), policy)
            source_se, source_sp = operating(src.y_true, src[name], t)
            external_se, external_sp = operating(ext.y_true, ext[name], t)
            target_t = cutoff(ext.y_true.to_numpy(int), ext[name].to_numpy(float), policy)
            apparent_se, apparent_sp = operating(ext.y_true, ext[name], target_t)
            rows.append(dict(unit=unit, model=name, source_target=policy,
                source_target_level=.9, source_threshold=t, source_sensitivity=source_se,
                source_specificity=source_sp, external_auc=roc_auc_score(ext.y_true,ext[name]),
                external_sensitivity=external_se, external_specificity=external_sp,
                external_optimized_threshold=target_t, external_optimized_sensitivity=apparent_se,
                external_optimized_specificity=apparent_sp))
    return rows


def plot_roc(ext: pd.DataFrame, policy: pd.DataFrame, out: Path):
    """Plot source-defined operating points without obscuring the underlying ROC curve."""
    names = {'Handcrafted': 'Handcrafted XGBoost',
             'Deep': 'EfficientNet-B0', 'Hybrid': 'Combined XGBoost'}
    fig, ax = plt.subplots(figsize=(7.25, 5.35))
    for model in MODELS:
        fpr, tpr, _ = roc_curve(ext.y_true, ext[model])
        auc = roc_auc_score(ext.y_true, ext[model])
        line, = ax.plot(fpr, tpr, lw=1.95,
                        label=f'{names[model]} (AUC {auc:.3f})')
        color = line.get_color()
        for target, mark in [('sensitivity', 'o'), ('specificity', 's')]:
            row = policy[(policy.model == model) &
                         (policy.source_target == target)].iloc[0]
            ax.plot(1-row.external_specificity, row.external_sensitivity, mark,
                    ms=7.5, color=color, markeredgecolor='black',
                    markeredgewidth=.7, zorder=5)
    ax.plot([0, 1], [0, 1], ls=':', lw=1, color='gray')
    ax.set(xlabel='1 - specificity', ylabel='Sensitivity',
           xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.15)
    ax.legend(loc='lower right', frameon=True, facecolor='white',
              framealpha=.95, fontsize=8.4)
    marker_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='black', ms=8, linestyle='None',
               label=r'Source sensitivity $\geq 0.90$'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='white',
               markeredgecolor='black', ms=8, linestyle='None',
               label=r'Source specificity $\geq 0.90$')]
    fig.legend(handles=marker_handles, loc='lower center',
               bbox_to_anchor=(.5, .015), ncol=2, frameon=False, fontsize=8.6)
    fig.subplots_adjust(left=.105, right=.97, top=.975, bottom=.195)
    fig.savefig(out/'roc_transport_lesion.pdf', bbox_inches='tight')
    fig.savefig(out/'roc_transport_lesion.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_boot(ci: pd.DataFrame, out: Path):
    """Show full-sample AUC points with stable-colored bootstrap intervals."""
    palette = {'naive_images': 'C0', 'clustered_lesions': 'C3'}
    labels = {'naive_images': 'Independent image resampling',
              'clustered_lesions': 'Whole-lesion resampling'}
    models = [('Deep', 'EfficientNet-B0'), ('Hybrid', 'Combined XGBoost')]
    fig, ax = plt.subplots(figsize=(6.75, 3.6))
    for j, (model, _) in enumerate(models):
        grp = ci[ci.model.eq(model)].set_index('method')
        for i, method in enumerate(['naive_images', 'clustered_lesions']):
            row = grp.loc[method]
            yy = j * 2.5 + i
            ax.plot([row.low, row.high], [yy, yy], lw=3.5,
                    color=palette[method], solid_capstyle='round')
            ax.plot(row.estimate, yy, 'o', ms=5,
                    color=palette[method], markeredgecolor='black', markeredgewidth=.35)
    ax.set_yticks([.5, 3.0], [name for _, name in models])
    ax.invert_yaxis()
    ax.set_xlabel('External image-level ROC-AUC (bootstrap 95% CI)')
    ax.grid(axis='x', alpha=.2)
    handles = [Line2D([0], [0], color=palette[k], marker='o', lw=3,
                      markersize=5, label=labels[k])
               for k in ('naive_images', 'clustered_lesions')]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, loc='lower center',
               bbox_to_anchor=(.5, .015), ncol=2)
    fig.subplots_adjust(left=.22, right=.975, top=.975, bottom=.265)
    fig.savefig(out/'bootstrap_comparison.pdf', bbox_inches='tight')
    fig.savefig(out/'bootstrap_comparison.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def run(repo: Path, out: Path, reps: int, seed: int):
    repo=repo.resolve(); out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    val=read_scores(repo/DATA/'ham_validation.csv')
    test=read_scores(repo/DATA/'ham_test.csv')
    ext=read_scores(repo/DATA/'bcn_external.csv')
    for a,b in [(val,test),(val,ext),(test,ext)]:
        if set(a.image_id) & set(b.image_id) or set(a.lesion_id) & set(b.lesion_id):
            raise ValueError('Partitions overlap')
    units={'image':(val,ext),'lesion':(lesions(val),lesions(ext))}
    result=pd.DataFrame([row for unit,(s,e) in units.items() for row in points(s,e,unit)])
    result.to_csv(out/'threshold_transport.csv',index=False,float_format='%.17g')
    lesion_rows=result[result.unit=='lesion'].copy()
    plot_roc(units['lesion'][1],lesion_rows,out)

    cis=[]
    for mod in ('Deep','Hybrid'):
        for method in ('naive_images','clustered_lesions'):
            low,high=auc_interval(ext,mod,method,reps,seed+len(cis))
            cis.append(dict(model=mod,method=method,estimate=float(roc_auc_score(ext.y_true, ext[mod])),low=low,high=high,width=high-low))
    ci=pd.DataFrame(cis);ci.to_csv(out/'bootstrap_comparison.csv',index=False)
    plot_boot(ci,out)

    intervals=[]
    for unit,(s,e) in units.items():
        for reselect in (False,True):
            lo,hi=nested_delta(s,e,'specificity',reps,reselect,seed+10+len(intervals))
            intervals.append(dict(unit=unit,policy='source_specificity_90',
                uncertainty_mode='source_reestimated' if reselect else 'fixed_cutoff',
                comparison='Hybrid_minus_Deep',metric='external_sensitivity',
                ci_low=lo,ci_high=hi,bootstrap_replicates=reps))
    pd.DataFrame(intervals).to_csv(out/'paired_bootstrap.csv',index=False)

    original=read_scores(repo/ORIGINAL/'ham_test.csv')
    corrected=test
    drops=[]
    for mod in MODELS:
        for name,subset in [('original',lesions(original)),('known_identity_corrected',lesions(corrected))]:
            drops.append(dict(model=mod,cohort='ham_test',evaluation=name,
                images=len(original) if name=='original' else len(corrected),lesions=len(subset),
                roc_auc=roc_auc_score(subset.y_true,subset[mod])))
    pd.DataFrame(drops).to_csv(out/'duplicate_sensitivity.csv',index=False)

    # Changes in output-score distributions across datasets within observed binary label.
    dist=[]
    for cohort,frame in [('ham_test',lesions(test)),('bcn_external',lesions(ext))]:
        for model in MODELS:
            for lab in (0,1):
                vals=frame.loc[frame.y_true.eq(lab),model].to_numpy(float)
                dist.append(dict(cohort=cohort,model=model,melanoma=int(lab),n_lesions=len(vals),
                    median=float(np.median(vals)),q25=float(np.quantile(vals,.25)),q75=float(np.quantile(vals,.75))))
    pd.DataFrame(dist).to_csv(out/'score_distributions.csv',index=False)

    src=lesions(val);target=lesions(ext)
    bins=[]
    for cohort,data in [('ham_validation',src),('bcn_external',target)]:
        for model in MODELS:
            # Rank by score, not by input order; equal-sized bins are descriptive.
            rank_order=np.argsort(data[model].to_numpy(),kind='stable')
            index_groups=np.array_split(rank_order,10)
            for b,ix in enumerate(index_groups):
                bins.append(dict(cohort=cohort,model=model,decile=b+1,n=len(ix),
                    mean_score=float(data[model].iloc[ix].mean()),
                    observed_melanoma_fraction=float(data.y_true.iloc[ix].mean())))
    pd.DataFrame(bins).to_csv(out/'calibration_deciles.csv',index=False)

    manifest={
      'origin':'squareshorts/melanoma-abc-hybrid, archive at 85e5140999a4c41764130f80a98c025f6f447e01',
      'seed':seed,'reps':reps,'source_dataset':'HAM10000 validation','external_dataset':'BCN20000',
      'source_images':len(val),'source_lesions':val.lesion_id.nunique(),
      'external_images':len(ext),'external_lesions':ext.lesion_id.nunique(),
      'data_modification':'none; scores supplied by archived fitted classifiers',
      'bootstrap_scope':'teaching demonstration; archive 2000 replicates authoritative',
      'interpretation':'target-derived cutoffs are apparent within-cohort optimizations and cannot validate transport'
    }
    (out/'run_summary.json').write_text(json.dumps(manifest,indent=2))
    print('WORKED EXAMPLE SUMMARY')
    print(lesion_rows[['model','source_target','source_threshold','external_auc','external_sensitivity','external_specificity','external_optimized_sensitivity','external_optimized_specificity']].to_string(index=False,float_format=lambda t:f'{t:.6f}'))
    print('BOOTSTRAP COMPARISON')
    print(ci.to_string(index=False,float_format=lambda t:f'{t:.6f}'))
    print('DUPLICATE SENSITIVITY')
    print(pd.DataFrame(drops).to_string(index=False,float_format=lambda t:f'{t:.6f}'))
    print('SCORE DISTRIBUTIONS')
    print(pd.DataFrame(dist).to_string(index=False,float_format=lambda t:f'{t:.6f}'))
    print('DONE')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True,help='Checkout of public melanoma-abc-hybrid archive')
    parser.add_argument('--out',type=Path,default=Path('tutorial_results'))
    parser.add_argument('--bootstrap',type=int,default=500)
    parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args()
    if args.bootstrap<100:
        parser.error('Use at least 100 bootstrap replicates')
    run(args.repo,args.out,args.bootstrap,args.seed)

if __name__=='__main__':
    main()