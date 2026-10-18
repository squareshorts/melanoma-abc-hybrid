"""One historical-specification reconstruction of the embeddings-only head.

The documented 2026-10-03 script specifies random_state=0 and n_jobs=4.
Fit once on original HAM training rows, save model and specification, and only
then evaluate its held-out and external predictions. No external optimization.
This creates a separately named reconstruction, not a replacement historical file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
if (HERE / "vendor").exists():
    sys.path.insert(0, str(HERE / "vendor"))

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

MODEL_NAME = "EmbeddingsOnly_Reconstructed"
PARAMETERS = {
    "n_estimators": 600,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "n_jobs": 4,
    "random_state": 0,
}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def input_record(path):
    return {"path": str(path), "bytes": Path(path).stat().st_size, "sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
        default=WORKSPACE / "tmp/project_history/extracted/melanoma_reanalysis_bundle")
    parser.add_argument("--historical-results", type=Path,
        default=WORKSPACE / "tmp/project_history/extracted/melanoma_reanalysis_results/melanoma_reanalysis_results")
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    out, bundle, historical = args.output, args.bundle, args.historical_results
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Output directory is nonempty. This entry point will not overwrite an earlier fit.")
    out.mkdir(parents=True, exist_ok=True)
    if xgb.__version__ != "3.2.0":
        raise RuntimeError("The frozen reconstruction requires xgboost==3.2.0")

    metadata_path = bundle / "external_metadata/HAM10000_metadata.csv"
    split_path = bundle / "data/splits/ham_lesion_split.json"
    embedding_path = bundle / "data/derived/embeddings/ham_effb0.npy"
    ids_path = bundle / "data/derived/embeddings/ham_effb0_ids.csv"
    original_script = HERE / "inputs/historical_embeddings_only_check.py"
    ham = pd.read_csv(metadata_path)
    ham["y_true"] = (ham.dx == "mel").astype(int)
    split = json.loads(split_path.read_text())
    embeddings = np.load(embedding_path, mmap_mode="r")
    embedding_ids = pd.read_csv(ids_path).image_id.astype(str).tolist()
    if len(embedding_ids) != len(set(embedding_ids)) or embeddings.shape[0] != len(embedding_ids):
        raise ValueError("HAM embedding identifiers are duplicated or misaligned")
    embedding_map = {image: i for i, image in enumerate(embedding_ids)}

    # Preserve the historical pandas row order and NumPy feature extraction.
    train = ham[ham.lesion_id.isin(set(split["train"]))][["image_id", "y_true"]].drop_duplicates()
    train = train[train.image_id.isin(embedding_map)].copy()
    x_train = np.asarray(embeddings[[embedding_map[i] for i in train.image_id]])
    y_train = train.y_true.to_numpy()
    if x_train.shape != (7000, 1280) or x_train.dtype != np.float32:
        raise ValueError("Recovered historical training tensor differs from its declared shape/dtype")
    if int(y_train.sum()) != 774:
        raise ValueError("Recovered training-label counts differ from the archived specification")
    lesion_map = ham.set_index("image_id").lesion_id.to_dict()
    train_rows = train.copy()
    train_rows["lesion_id"] = train_rows.image_id.map(lesion_map)
    train_rows["embedding_row"] = train_rows.image_id.map(embedding_map)
    train_rows.insert(0, "training_row", np.arange(len(train_rows)))
    train_rows.to_csv(out / "training_rows.csv", index=False)
    np.save(out / "training_features_float32.npy", x_train)
    np.save(out / "training_labels.npy", y_train)

    specification = {
        "model_name": MODEL_NAME,
        "specification_frozen_utc": datetime.now(timezone.utc).isoformat(),
        "status": "historical-specification reconstruction; original embeddings-only fitted object unavailable",
        "historical_specification_source": input_record(original_script),
        "historical_parameters": PARAMETERS,
        "training_rule": "Original HAM metadata row order; train lesion membership; select image_id/y_true; drop_duplicates; filter to archived embedding IDs; indexed np.asarray extraction with no dtype conversion.",
        "training_feature_dtype": str(x_train.dtype),
        "training_label_dtype": str(y_train.dtype),
        "training_shape": list(x_train.shape),
        "training_positive_count": int(y_train.sum()),
        "software": {"python": platform.python_version(), "xgboost": xgb.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__},
        "inputs_used_before_fitting": [input_record(p) for p in
            (metadata_path, split_path, embedding_path, ids_path)],
        "frozen_training_exports": [input_record(out / p) for p in
            ("training_rows.csv", "training_features_float32.npy", "training_labels.npy")],
        "number_of_fits_in_this_run": 1,
        "selection": "No validation selection, external tuning, alternative seed, hyperparameter search or CNN retraining.",
        "environment_qualification": "XGBoost matches the recovered later audit lock (3.2.0). Other current runtime package versions are recorded above; the exact original October 3 runtime was not separately recovered.",
    }
    write_json(out / "predeclared_specification.json", specification)
    print("Frozen historical seed-0 specification; fitting one head.", flush=True)
    model = xgb.XGBClassifier(**PARAMETERS)
    model.fit(x_train, y_train)
    model_path = out / "embeddings_only_reconstructed.ubj"
    model.save_model(model_path)
    (out / "booster_configuration.json").write_text(model.get_booster().save_config() + "\n")
    model_frozen = {"saved_before_external_evaluation_utc": datetime.now(timezone.utc).isoformat(),
                    "model_name": MODEL_NAME, "model_file": input_record(model_path),
                    "predeclared_specification_sha256": digest(out / "predeclared_specification.json")}
    write_json(out / "model_frozen_manifest.json", model_frozen)
    print("Model saved. Generating held-out predictions and evaluating the frozen fit.", flush=True)

    metric_rows, prediction_files = [], []

    def export_predictions(frame, features, partition):
        frame = frame.copy()
        frame["y_prob"] = model.predict_proba(features)[:, 1]
        frame["model"] = MODEL_NAME
        frame["partition"] = partition
        filename = "predictions_" + partition + ".csv"
        frame.to_csv(out / filename, index=False, float_format="%.17g")
        prediction_files.append(input_record(out / filename))
        metric_rows.append({"model": MODEL_NAME, "partition": partition,
            "images": len(frame), "lesions": frame.lesion_id.nunique(),
            "melanoma_images": int(frame.y_true.sum()),
            "AUC": float(roc_auc_score(frame.y_true, frame.y_prob)),
            "average_precision": float(average_precision_score(frame.y_true, frame.y_prob)),
            "Brier": float(brier_score_loss(frame.y_true, frame.y_prob))})
        return frame

    for partition in ("val", "test"):
        frame = ham[ham.lesion_id.isin(set(split[partition]))][["image_id", "lesion_id", "y_true"]].drop_duplicates()
        frame = frame[frame.image_id.isin(embedding_map)].copy()
        frame["embedding_row"] = frame.image_id.map(embedding_map)
        features = np.asarray(embeddings[[embedding_map[i] for i in frame.image_id]])
        export_predictions(frame, features, "ham_" + partition)

    external_embedding_path = bundle / "data/derived/embeddings/bcn20000_effb0.npy"
    external_ids_path = bundle / "data/derived/embeddings/bcn20000_effb0_ids.csv"
    external_evaluation_path = historical / "bcn_predictions_metadata_clustered.csv"
    external_embeddings = np.load(external_embedding_path, mmap_mode="r")
    external_ids = pd.read_csv(external_ids_path).image_id.astype(str).tolist()
    if len(external_ids) != len(set(external_ids)) or len(external_ids) != external_embeddings.shape[0]:
        raise ValueError("External embedding identifiers are duplicated or misaligned")
    external_map = {image: i for i, image in enumerate(external_ids)}
    external = pd.read_csv(external_evaluation_path)
    frame = external[external.image_id.isin(external_map)][["image_id", "lesion_id", "y_true", "Hybrid"]].copy()
    if len(frame) != 12413 or frame.image_id.nunique() != 12413 or frame.lesion_id.nunique() != 3576:
        raise ValueError("External evaluation cohort differs from the archived historical analysis")
    frame["embedding_row"] = frame.image_id.map(external_map)
    external_features = np.asarray(external_embeddings[[external_map[i] for i in frame.image_id]])
    predicted = export_predictions(frame.drop(columns="Hybrid"), external_features, "bcn_external")
    hybrid_auc = float(roc_auc_score(frame.y_true, frame.Hybrid))
    reconstructed_auc = float(roc_auc_score(predicted.y_true, predicted.y_prob))
    # The archived benchmark is opened only after the fixed model and scores exist.
    old_table_path = historical / "table_external_embeddings_only_ablation.csv"
    old_table = pd.read_csv(old_table_path)
    old_rows = old_table[old_table.model.astype(str).str.lower().str.contains("embeddings")]
    if len(old_rows) != 1:
        raise ValueError("Cannot identify the archived embeddings-only comparison row")
    archived_auc = float(old_rows.iloc[0].AUC)
    reproduction = {
        "model_name": MODEL_NAME,
        "reconstructed_external_AUC": reconstructed_auc,
        "archived_external_AUC": archived_auc,
        "difference_from_archived_AUC": reconstructed_auc - archived_auc,
        "matches_archived_AUC_at_absolute_tolerance_1e_minus_12": abs(reconstructed_auc - archived_auc) <= 1e-12,
        "hybrid_external_AUC_same_rows": hybrid_auc,
        "hybrid_minus_reconstructed_external_AUC": hybrid_auc - reconstructed_auc,
        "qualification": "An AUC match is an aggregate reproduction result, not proof of row-wise identity to unavailable original embeddings-only scores. Model remains explicitly labeled reconstructed.",
        "after_evaluation_action": "No alternative parameterization or refit is selected based on these external results.",
    }
    pd.DataFrame(metric_rows).to_csv(out / "reconstructed_performance.csv", index=False, float_format="%.17g")
    write_json(out / "reproduction_comparison.json", reproduction)
    write_json(out / "final_artifact_manifest.json", {
        "model": model_frozen, "prediction_files": prediction_files,
        "evaluation_input_files": [input_record(p) for p in
            (external_embedding_path, external_ids_path, external_evaluation_path, old_table_path)],
        "performance_comparison": reproduction,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps(reproduction, indent=2), flush=True)


if __name__ == "__main__":
    main()
