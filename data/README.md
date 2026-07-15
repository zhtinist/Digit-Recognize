# Datasets

The MNIST-format CSV files used by this project come from the Kaggle
[Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)
competition. They are **not** committed to this repository (see `.gitignore`).

Download `train.csv`, `test.csv`, and `sample_submission.csv` from Kaggle and
place them here:

```
data/
├── raw/
│   ├── train.csv              # 42,000 labelled 28x28 images
│   ├── test.csv               # 28,000 unlabelled images
│   └── sample_submission.csv
└── processed/                 # generated submissions land here
```

### Format

* `train.csv` — first column `label` (0-9), followed by 784 pixel columns
  (`pixel0` … `pixel783`), each an integer in `[0, 255]`.
* `test.csv` — 784 pixel columns only.
