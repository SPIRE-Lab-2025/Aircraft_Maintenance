# Privacy-Preserving Cross-Operator Learning for Aircraft Engine Prognostics

**AIAA DASC 2026** · Kanchon Gharami, Shafika Showkat Moni · Embry-Riddle Aeronautical University

A unified, physics-guided federated learning framework for joint early fault detection and remaining useful life (RUL) estimation across multi-operator aircraft engine fleets — without sharing raw sensor data.

---

## Overview

- **TCN-based dual-head architecture** for joint fault detection and RUL quantile estimation
- **Physics-guided monotonic constraint** enforcing non-increasing RUL trajectories
- **Federated learning with trimmed-mean aggregation** for robust cross-operator training under non-IID data
- **Calibrated uncertainty** via quantile regression (Q10/Q50/Q90)
- Evaluated on **NASA C-MAPSS** and **NASA PHM 2008** benchmarks

---

## Repository Structure

```
Aircraft_Maintenance/
├── CMAPSS/
│   ├── CMAPSS_v3.ipynb            # Full experiment suite (centralized + FL, IEEE plots)
│   └── dataset/CMAPSSData/        # NASA C-MAPSS txt files (FD001–FD004) + RUL labels
│
└── PHM2008/
    ├── PHM08_main_pipeline.ipynb  # Centralized training and evaluation
    ├── PHM08_FL_pipeline.ipynb    # Federated learning pipeline
    ├── requirements.txt
    └── dataset/                   # PHM 2008 Challenge files (train/test/final_test)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/aircraft-engine-prognostics.git
cd aircraft-engine-prognostics
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r PHM2008/requirements.txt
```

> **GPU note:** The requirements file pins `torch==2.5.1+cu121` (CUDA 12.1). If your system uses a different CUDA version or CPU only, replace the torch/torchvision/torchaudio lines accordingly. See [PyTorch installation](https://pytorch.org/get-started/locally/).

### 4. Launch Jupyter

```bash
jupyter notebook
```

---

## Datasets

### NASA C-MAPSS

The dataset files are included under `CMAPSS/dataset/CMAPSSData/`:

```
train_FD001.txt  test_FD001.txt  RUL_FD001.txt
train_FD002.txt  test_FD002.txt  RUL_FD002.txt
train_FD003.txt  test_FD003.txt  RUL_FD003.txt
train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

### NASA PHM 2008 Challenge

Dataset files are included under `PHM2008/dataset/`:

```
train.txt   test.txt   final_test.txt
```

Both datasets consist of multivariate run-to-failure time series from turbofan engines. See the `readme.txt` files in each dataset folder for full column descriptions.

---

## Running Experiments

### C-MAPSS (Centralized + Federated)

Open `CMAPSS/CMAPSS_v3.ipynb`. The notebook is self-contained and produces:

- IID vs. non-IID heterogeneity stress tests (condition-skew, severity-skew, quantity-skew, label-skew, stage-skew)
- FedAvg vs. TrimmedMean vs. Median aggregation comparison
- Physics ablation (monotonic loss ON/OFF)
- Calibration reliability diagrams, sharpness–coverage Pareto plots
- Early fault timing: τ sweep over {10, 20, 30, 40}, recall heatmaps
- Latent PCA trajectories and cross-client alignment heatmaps

Results (CSV/JSON metrics + PDF plots) are saved to `CMAPSS/results/`.

> Start with the **Smoke Test** cell for a fast sanity check before running the full suite.

### PHM 2008 (Centralized)

Open `PHM2008/PHM08_main_pipeline.ipynb`. Runs the dual-head DS-TCN pipeline with physics-guided monotonic constraint. Results saved to `PHM2008/results/phm08/`.

### PHM 2008 (Federated)

Open `PHM2008/PHM08_FL_pipeline.ipynb`. Simulates federated training by splitting engine units across clients. Per-round validation curves, per-client metrics, and predictions are saved to `PHM2008/results/phm08_fl/`.

---

## Key Results

| Dataset | Setting | Fault AUPRC | Recall@FPR=1% | RUL MAE | RUL RMSE |
|---|---|---|---|---|---|
| C-MAPSS FD004 | Centralized | 0.418 | 0.317 | 12.14 | 19.93 |
| C-MAPSS FD004 | Federated (TrimmedMean) | 0.436 | 0.366 | 13.97 | 20.34 |
| PHM 2008 | Centralized | 0.938 | 0.704 | 25.38 | 33.63 |
| PHM 2008 | Federated | 0.863 | 0.644 | 33.97 | 46.15 |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{gharami2026privacypreserving,
  title     = {Privacy Preserving Cross-Operator Learning for Aircraft Engine Degradation and RUL Estimation},
  author    = {Gharami, Kanchon and Moni, Shafika Showkat},
  booktitle = {Proceedings of the 45th IEEE/AIAA Digital Avionics Systems Conference (DASC)},
  year      = {2026}
}
```

---

## License

This repository is for academic research purposes. Dataset files are distributed under their original NASA open-data terms. See individual `readme.txt` files in each dataset folder.
