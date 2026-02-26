# Data Instructions

This repository does not redistribute datasets.

To reproduce experiments:

1. Download HAM10000 from:
   https://doi.org/10.1038/sdata.2018.161

2. Download ISIC 2018 Task 1 and Task 3 from:
   https://challenge.isic-archive.com

3. Place datasets under:

data/raw/HAM10000/
data/raw/ISIC2018/

4. Run:

python -m experiments.01_make_split
python -m experiments.02_segment_isic_task1
...