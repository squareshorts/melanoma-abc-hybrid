#!/usr/bin/env python3
"""Frozen EfficientNet-B0 representation-projection pilot.

This script never loads images or CNN weights and never retrains a CNN.
Selection is limited to HAM10000 validation; test/BCN data are read afterward.
Existing original split and identity-corrected evaluation files are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

SEED = 20261109
REPRO = Path('reproducibility/threshold_transport_v2_1_0')
TRAIN_PATH = REPRO / 'comparator/outputs/training_rows.csv'
VAL_PATH = REPRO / 'inputs/four_model_known_identity_corrected/ham_validation.csv'
TEST_PATH = REPRO / 'inputs/four_model_known_identity_corrected/ham_test.csv'
BCN_PATH = REPRO / 'inputs/four_model_known_identity_corrected/bcn_external.csv'
DUP_PATH = REPRO / 'provenance/outputs/ham_known_duplicate_validation_test_exclusions.csv'
EMB_DIR = Path('data/derived/embeddings')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_frame(path: Path, split: str, exclusions: pd.DataFrame | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    required = {'image_id', 'lesion_id', 'y_true'}
    if not required.issubset(df):
        raise ValueError(f'{path}: missing {required - set(df)}')
    df = df[['image_id', 'lesion_id', 'y_true']].copy()
    df['image_id'] = df['image_id'].astype(str)
    df['lesion_id'] = df['lesion_id'].astype(str)
    df['y_true'] = pd.to_numeric(df['y_true'], errors='raise').astype(int)
    if df[['image_id', 'lesion_id']].isna().any().any() or df['y_true'].isna().any():
        raise ValueError(f'{split}: missing identifiers/labels')
    if not set(df.y_true.unique()).issubset({0, 1}) or df.y_true.nunique() != 2:
        raise ValueError(f'{split}: invalid binary target')
    if not df.image_id.is_unique:
        raise ValueError(f'{split}: repeated image_id')
    if (df.groupby('lesion_id').y_true.nunique() > 1).any():
        raise ValueError(f'{split}: inconsistent labels within lesion')
    if exclusions is not None:
        bad = set(exclusions.loc[exclusions['partition'].eq(split), 'lesion_id'].astype(str))
        df = df.loc[~df.lesion_id.isin(bad)].copy()
    return df.reset_index(drop=True)


def load_embeddings(root: Path, cohort: str, df: pd.DataFrame) -> np.ndarray:
    epath = root / EMB_DIR / f'{cohort}_effb0.npy'
    ipath = root / EMB_DIR / f'{cohort}_effb0_ids.csv'
    ids = pd.read_csv(ipath)['image_id'].astype(str)
    if not ids.is_unique:
        raise ValueError(f'{cohort}: duplicate embedding IDs')
    mat = np.load(epath, mmap_mode='r', allow_pickle=False)
    if mat.ndim != 2 or len(mat) != len(ids) or mat.shape[1] != 1280:
        raise ValueError(f'{cohort}: incorrect 1280-D embedding shape / IDs: {mat.shape}')
    loc = pd.Series(np.arange(len(ids)), index=ids.values)
    ix = loc.reindex(df.image_id.values)
    if ix.isna().any():
        raise ValueError(f'{cohort}: {int(ix.isna().sum())} requested images lack frozen embeddings')
    output = np.asarray(mat[ix.to_numpy(dtype=int)], dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError(f'{cohort}: nonfinite embeddings')
    return output


def balanced_lesion_weights(df: pd.DataFrame) -> np.ndarray:
    sizes = df.groupby('lesion_id').image_id.transform('size').to_numpy()
    n_lesion_classes = df.groupby('lesion_id').y_true.first().value_counts()
    y = df.y_true.to_numpy()
    w = 1.0 / sizes / np.array([n_lesion_classes[v] for v in y], dtype=float)
    return w * len(w) / w.sum()


def group_residuals(z: np.ndarray, lesions: pd.Series):
    gi, keys = pd.factorize(lesions.astype(str), sort=True)
    size = np.bincount(gi).astype(float)
    sums = np.zeros((len(keys), z.shape[1]), dtype=np.float64)
    np.add.at(sums, gi, z)
    means = (sums / size[:, None]).astype(np.float32)
    centered = z - means[gi]
    # Each multi-image lesion gets roughly equal weight, independently of view count.
    weighted = centered / np.sqrt(np.maximum(size[gi] - 1., 1.))[:, None]
    total = z - z.mean(axis=0, keepdims=True)
    return weighted.astype(np.float32), total.astype(np.float32), int((size >= 2).sum())


def normalized_within_fraction(w: np.ndarray, t: np.ndarray, u: np.ndarray | None, k: int) -> tuple[float, float, float]:
    wn = float(np.sum(w * w, dtype=np.float64))
    tn = float(np.sum(t * t, dtype=np.float64))
    if u is not None and k:
        basis = u[:, :k]
        wn -= float(np.sum((w @ basis) ** 2, dtype=np.float64))
        tn -= float(np.sum((t @ basis) ** 2, dtype=np.float64))
    return wn / tn, wn, tn


def projection(z: np.ndarray, u: np.ndarray | None, k: int) -> np.ndarray:
    if u is None or k == 0:
        return z
    basis = u[:, :k]
    return np.asarray(z - (z @ basis) @ basis.T, dtype=np.float32)


def lesion_table(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    x = df[['image_id', 'lesion_id', 'y_true']].copy()
    x['probability'] = scores
    result = x.groupby('lesion_id', sort=True).agg(
        y_true=('y_true', 'first'), probability=('probability', 'mean'),
        images=('image_id', 'size')).reset_index()
    return result


def specificity_threshold(val: pd.DataFrame, target: float = 0.9) -> float:
    neg = np.sort(val.loc[val.y_true.eq(0), 'probability'].to_numpy())
    if len(neg) < 10:
        raise ValueError('Insufficient negative validation lesions for specificity threshold')
    # Predict positive for score > cutoff; at least the targeted source specificity.
    return float(neg[min(len(neg) - 1, int(np.ceil(target * len(neg))) - 1)])


def evaluate(lesion: pd.DataFrame, threshold: float) -> dict:
    yy = lesion.y_true.to_numpy(dtype=int)
    pp = lesion.probability.to_numpy(dtype=float)
    classified = pp > threshold
    pos = yy == 1
    neg = yy == 0
    return dict(lesions=int(len(lesion)), positive_lesions=int(pos.sum()),
                roc_auc=float(roc_auc_score(yy, pp)),
                average_precision=float(average_precision_score(yy, pp)),
                sensitivity=float(classified[pos].mean()),
                specificity=float((~classified[neg]).mean()),
                tp=int(classified[pos].sum()), fn=int((~classified[pos]).sum()),
                tn=int((~classified[neg]).sum()), fp=int(classified[neg].sum()))


def auc_diff_bootstrap(a: pd.DataFrame, b: pd.DataFrame, replicates: int, seed: int) -> dict:
    p = a[['lesion_id', 'y_true', 'probability']].merge(
        b[['lesion_id', 'y_true', 'probability']], on=['lesion_id', 'y_true'],
        suffixes=('_candidate', '_baseline'), validate='one_to_one')
    if len(p) != len(a) or len(p) != len(b):
        raise ValueError('Unmatched paired lesion scores in bootstrap')
    yy = p.y_true.to_numpy(dtype=int)
    c = p.probability_candidate.to_numpy(dtype=float)
    base = p.probability_baseline.to_numpy(dtype=float)
    estimate = float(roc_auc_score(yy, c) - roc_auc_score(yy, base))
    pos = np.flatnonzero(yy == 1)
    neg = np.flatnonzero(yy == 0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for i in range(replicates):
        indices = np.r_[rng.choice(pos, len(pos), replace=True),
                        rng.choice(neg, len(neg), replace=True)]
        deltas[i] = roc_auc_score(yy[indices], c[indices]) - roc_auc_score(yy[indices], base[indices])
    ci = np.percentile(deltas, [2.5, 97.5])
    return dict(delta_auc=estimate, ci_low=float(ci[0]), ci_high=float(ci[1]),
                bootstrap_replicates=replicates, unit='lesion', stratified=True)


def run(root: Path, out: Path, ranks: list[int], bootstraps: int) -> dict:
    root, out = root.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dup = pd.read_csv(root / DUP_PATH)
    train = read_frame(root / TRAIN_PATH, 'train')
    val = read_frame(root / VAL_PATH, 'val', dup)
    test = read_frame(root / TEST_PATH, 'test', dup)
    for x, y in [(train, val), (train, test), (val, test)]:
        if set(x.image_id) & set(y.image_id) or set(x.lesion_id) & set(y.lesion_id):
            raise ValueError('HAM partition image/lesion overlap')
    z_train_raw = load_embeddings(root, 'ham', train)
    z_val_raw = load_embeddings(root, 'ham', val)
    z_test_raw = load_embeddings(root, 'ham', test)
    # Training-only normalization, reused unchanged for HAM and BCN.
    w_train = balanced_lesion_weights(train)
    scaler = StandardScaler().fit(z_train_raw, sample_weight=w_train)
    z_train = scaler.transform(z_train_raw).astype(np.float32)
    z_val = scaler.transform(z_val_raw).astype(np.float32)
    z_test = scaler.transform(z_test_raw).astype(np.float32)
    del z_train_raw, z_val_raw, z_test_raw
    w_res, t_res, n_multi = group_residuals(z_train, train.lesion_id)
    max_rank = max(ranks)
    if max_rank >= z_train.shape[1]:
        raise ValueError('Requested removal rank is too high')
    _, _, v_within = randomized_svd(w_res, n_components=max_rank, random_state=SEED,
                                     n_iter=4, n_oversamples=16)
    _, _, v_total = randomized_svd(t_res, n_components=max_rank, random_state=SEED + 1,
                                    n_iter=4, n_oversamples=16)
    bases: dict[str, np.ndarray | None] = {
        'unprojected': None, 'within_lesion': v_within.T.astype(np.float32),
        'ordinary_pca': v_total.T.astype(np.float32)
    }
    rng = np.random.default_rng(SEED + 2)
    random_matrix = rng.standard_normal((z_train.shape[1], max_rank)).astype(np.float32)
    q, _ = np.linalg.qr(random_matrix, mode='reduced')
    bases['random_subspace'] = q.astype(np.float32)
    configs = [('unprojected', 0)] + [(name, k) for name in
              ('within_lesion', 'ordinary_pca', 'random_subspace') for k in ranks]
    models, val_tables, thresholds, records = {}, {}, {}, []
    baseline_val_auc = None
    var_base = normalized_within_fraction(w_res, t_res, None, 0)[0]
    for name, k in configs:
        ident = name if k == 0 else f'{name}_k{k}'
        z_fit = projection(z_train, bases[name], k)
        head = LogisticRegression(C=1., penalty='l2', solver='liblinear', max_iter=1000,
                                  tol=1e-4, random_state=SEED)
        head.fit(z_fit, train.y_true.to_numpy(), sample_weight=w_train)
        del z_fit
        v_scores = head.predict_proba(projection(z_val, bases[name], k))[:, 1]
        tab = lesion_table(val, v_scores)
        threshold = specificity_threshold(tab)
        met = evaluate(tab, threshold)
        within_frac, vwithin, vtotal = normalized_within_fraction(w_res, t_res, bases[name], k)
        metrics = dict(model=ident, kind=name, removed_rank=k, cohort='ham_val',
                       threshold_from_ham_val=threshold,
                       within_fraction_train=within_frac,
                       within_fraction_relative_to_baseline=within_frac / var_base,
                       within_energy_retained_train=vwithin / float(np.sum(w_res ** 2, dtype=np.float64)),
                       total_energy_retained_train=vtotal / float(np.sum(t_res ** 2, dtype=np.float64)),
                       **met)
        records.append(metrics)
        if name == 'unprojected':
            baseline_val_auc = met['roc_auc']
        models[ident] = (head, bases[name], k)
        val_tables[ident] = tab
        thresholds[ident] = threshold
    assert baseline_val_auc is not None
    # Source validation only: most nuisance removal subject to <=0.01 AUC loss.
    candidates = [r for r in records if r['kind'] == 'within_lesion' and
                  r['roc_auc'] >= baseline_val_auc - 0.01]
    chosen = min(candidates, key=lambda r: (r['within_fraction_train'], -r['roc_auc'], r['removed_rank'])) if candidates else None
    chosen_id = chosen['model'] if chosen else None
    # External cohort is loaded only after all fits and rank selection are frozen.
    bcn = read_frame(root / BCN_PATH, 'bcn')
    if set(bcn.image_id) & set(train.image_id) | set(bcn.image_id) & set(val.image_id) | set(bcn.image_id) & set(test.image_id):
        raise ValueError('Cross-dataset image ID overlap')
    z_bcn = scaler.transform(load_embeddings(root, 'bcn20000', bcn)).astype(np.float32)
    bcn_w, bcn_t, bcn_multiview = group_residuals(z_bcn, bcn.lesion_id)
    test_tables, bcn_tables = {}, {}
    predictions = []
    for name, k in configs:
        ident = name if k == 0 else f'{name}_k{k}'
        model, basis, rk = models[ident]
        for subset, x, df, target in [('ham_val', z_val, val, val_tables),
                                      ('ham_test', z_test, test, test_tables),
                                      ('bcn_external', z_bcn, bcn, bcn_tables)]:
            if subset == 'ham_val':
                lesion = target[ident]
            else:
                scores = model.predict_proba(projection(x, basis, rk))[:, 1]
                lesion = lesion_table(df, scores)
                target[ident] = lesion
                met = evaluate(lesion, thresholds[ident])
                bcn_frac = normalized_within_fraction(bcn_w, bcn_t, basis, rk)[0] if subset == 'bcn_external' else np.nan
                records.append(dict(model=ident, kind=name, removed_rank=k, cohort=subset,
                                    threshold_from_ham_val=thresholds[ident],
                                    within_fraction_train=next(r['within_fraction_train'] for r in records if r['model'] == ident and r['cohort'] == 'ham_val'),
                                    within_fraction_relative_to_baseline=next(r['within_fraction_relative_to_baseline'] for r in records if r['model'] == ident and r['cohort'] == 'ham_val'),
                                    within_fraction_bcn=bcn_frac,
                                    within_fraction_bcn_relative_to_baseline=(bcn_frac / normalized_within_fraction(bcn_w, bcn_t, None, 0)[0]
                                                                            if subset == 'bcn_external' else np.nan),
                                    **met))
            tmp = lesion[['lesion_id', 'y_true', 'images', 'probability']].copy()
            tmp.insert(0, 'cohort', subset)
            tmp.insert(0, 'model', ident)
            predictions.append(tmp)
    result_frame = pd.DataFrame.from_records(records)
    result_frame.to_csv(out / 'model_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / 'lesion_predictions.csv', index=False)
    comparisons = []
    for cohort_name, tables in [('ham_test', test_tables), ('bcn_external', bcn_tables)]:
        for name in [n for n in tables if n != 'unprojected']:
            if chosen_id and name not in {chosen_id,
                f'ordinary_pca_k{chosen["removed_rank"]}',
                f'random_subspace_k{chosen["removed_rank"]}'}:
                continue
            ci = auc_diff_bootstrap(tables[name], tables['unprojected'], bootstraps,
                                    seed=SEED + (0 if cohort_name == 'ham_test' else 1000) + len(comparisons))
            comparisons.append(dict(cohort=cohort_name, model=name, reference='unprojected', **ci))
    pd.DataFrame(comparisons).to_csv(out / 'paired_auc_deltas.csv', index=False)
    np.savez_compressed(out / 'pilot_model_state.npz', scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        basis_within=bases['within_lesion'], basis_pca=bases['ordinary_pca'],
                        basis_random=bases['random_subspace'],
                        **{f'coef_{i}': models[n][0].coef_ for i, (n, _) in enumerate(models.items())},
                        **{f'intercept_{i}': models[n][0].intercept_ for i, (n, _) in enumerate(models.items())})
    inputs = [TRAIN_PATH, VAL_PATH, TEST_PATH, BCN_PATH, DUP_PATH,
              EMB_DIR / 'ham_effb0.npy', EMB_DIR / 'ham_effb0_ids.csv',
              EMB_DIR / 'bcn20000_effb0.npy', EMB_DIR / 'bcn20000_effb0_ids.csv']
    provenance = {str(p): digest(root / p) for p in inputs}
    provenance['script_sha256'] = digest(Path(__file__))
    ci_chosen = {c['cohort']: c for c in comparisons if c['model'] == chosen_id} if chosen else {}
    if chosen:
        by = {(r['cohort'], r['model']): r for r in records}
        difft = by['ham_test', chosen_id]['roc_auc'] - by['ham_test', 'unprojected']['roc_auc']
        diffb = by['bcn_external', chosen_id]['roc_auc'] - by['bcn_external', 'unprojected']['roc_auc']
        ctrl = next(r for r in records if r['model'] == f'random_subspace_k{chosen["removed_rank"]}' and r['cohort'] == 'ham_val')
        # Bounded go/no-go criterion, not a hypothesis test of clinical equivalence.
        go = bool(chosen['within_fraction_relative_to_baseline'] <= 0.90 and
                  chosen['within_fraction_relative_to_baseline'] < ctrl['within_fraction_relative_to_baseline'] - 0.05 and
                  difft >= -0.02 and diffb >= -0.02 and
                  ci_chosen['bcn_external']['ci_low'] > -0.03)
    else:
        go = False
        difft = diffb = None
    decision = dict(experiment='frozen_effb0_multiview_projection_pilot', seed=SEED,
                    projection='remove leading equal-lesion-weighted within-lesion residual principal axes',
                    classifier='class/lesion-balanced logistic ridge C=1 on fixed HAM training embeddings',
                    rank_grid=ranks, validation_auc_loss_tolerance=0.01,
                    selection='highest within-lesion scatter reduction under HAM validation lesion-AUC tolerance',
                    train_images=len(train), val_images=len(val), test_images=len(test), bcn_images=len(bcn),
                    train_lesions=train.lesion_id.nunique(), val_lesions=val.lesion_id.nunique(),
                    test_lesions=test.lesion_id.nunique(), bcn_lesions=bcn.lesion_id.nunique(),
                    multi_image_train_lesions=n_multi, multi_image_bcn_lesions=bcn_multiview,
                    selected_model=chosen_id, selected_rank=chosen['removed_rank'] if chosen else None,
                    source_test_delta_auc=difft, external_delta_auc=diffb,
                    selected_paired_auc_intervals=ci_chosen,
                    go_for_distinct_methods_manuscript=go,
                    interpretive_limit='within-lesion image variation is a proxy, not proof of causal viewpoint or biological mechanisms',
                    head_label='Melanoma vs non-melanoma',
                    source_version='corrected frozen threshold-transport inputs; historic train rows retained',
                    removed_known_cross_partition_duplicate_groups=True,
                    software={'python':platform.python_version(), 'numpy':np.__version__,
                              'pandas':pd.__version__, 'scipy':scipy.__version__, 'scikit_learn':sklearn.__version__},
                    provenance_sha256=provenance)
    with (out / 'pilot_decision.json').open('w', encoding='utf8') as f:
        json.dump(decision, f, indent=2)
    with (out / 'README.md').open('w', encoding='utf8') as f:
        f.write('# Frozen EfficientNet representation projection pilot\n\n'
                'All heads are fitted solely on HAM train; model/rank selection uses HAM validation. '
                'HAM test and BCN20000 external data are held out from selection. '\
                'The same source-derived validation cutoff is transferred unchanged to both cohorts.\n\n'
                'Embeddings are unchanged by CNN training. Scores in lesion_predictions.csv are lesion-mean probabilities. '
                'Known cross-partition HAM duplicate groups are excluded from validation/test only; '
                'the original train observations remain untouched.\n\n'
                'The method uses repeated images of a lesion as a proxy for acquisition/view-related variation. '
                'It cannot disentangle camera angle, illumination, morphology, time, or artifact sources.\n\n'
                'A go decision is a deliberately strict pilot screen, not a clinical equivalence claim. '
                'The previously reported score-level fusion analysis is independent and is not rerun or overwritten.\n')
    print('PILOT_DECISION_JSON_BEGIN', flush=True)
    print(json.dumps({k: decision[k] for k in ['selected_model', 'selected_rank',
          'source_test_delta_auc','external_delta_auc','go_for_distinct_methods_manuscript',
          'train_images','val_images','test_images','bcn_images',
          'multi_image_train_lesions','multi_image_bcn_lesions','selected_paired_auc_intervals']}, indent=2), flush=True)
    print('PILOT_DECISION_JSON_END', flush=True)
    print('MODEL_METRICS_BEGIN', flush=True)
    cols=['cohort','model','removed_rank','roc_auc','sensitivity','specificity',
          'within_fraction_relative_to_baseline','within_fraction_bcn_relative_to_baseline']
    print(result_frame[cols].sort_values(['cohort','model']).to_csv(index=False), flush=True)
    print('MODEL_METRICS_END', flush=True)
    return decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out', type=Path, default=Path('results/representation_projection_pilot'))
    ap.add_argument('--ranks', nargs='+', type=int, default=[16, 32, 64, 128])
    ap.add_argument('--bootstrap', type=int, default=500)
    a = ap.parse_args()
    if a.bootstrap < 100 or len(set(a.ranks)) != len(a.ranks) or min(a.ranks) < 1:
        ap.error('Need >=100 bootstraps and distinct positive ranks')
    run(a.root, a.out, sorted(a.ranks), a.bootstrap)


if __name__ == '__main__':
    main()
