# Data instructions

Raw image datasets are not redistributed by this repository. Download them from their original sources and set `data_root` in `config.yaml` to the parent directory containing the datasets.

The current manuscript uses HAM10000 for model development, ISIC 2018 Task 1 for segmentation and saliency-mask benchmarking, and BCN20000 for external evaluation and lesion-level clustering metadata. External melanoma labels are matched by image identifier from the ISIC 2019 ground-truth table used by the original BCN evaluation scripts.

Example local layout:

```text
C:/work/datasets/
  HAM10000/
    images/
    metadata.csv
  ISIC2018/
    Task1/
      images/
      masks/
    Task3/
      images/
      labels.csv
  BCN20000/
    images/
      metadata.csv
      ISIC_*.jpg
```

The corrected external audit expects locally generated prediction files under `results/runs/` and writes current tables to `results/final/`:

```powershell
.\.venv\Scripts\python.exe experiments\23_final_external_audit.py
```

Operating thresholds are selected from HAM10000 validation predictions and applied unchanged to BCN20000. External uncertainty is resampled by BCN20000 `lesion_id`.
