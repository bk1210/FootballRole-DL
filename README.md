# ⚽ FootballRole-DL — IoT Wearable Player Position Role Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-yellow?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU-20BEFF?style=for-the-badge&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Classifies football players into positional roles — Attacker, Midfielder, Defender — directly from multi-body IoT wearable sensor streams. No GPS, no video, no manual labels.**

[Features](#-features) • [How It Works](#-how-it-works) • [Results](#-results) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Tech Stack](#-tech-stack) • [Contact](#-contact)

</div>

---

## 📌 Overview

**FootballRole-DL** goes beyond generic activity recognition to answer the question that actually matters in coaching: *Is this player moving like an Attacker, a Midfielder, or a Defender right now?*

The key insight is that each positional role has a distinct biomechanical signature — explosive ankle impulses for Attackers, sustained chest acceleration variability for Midfielders, controlled wrist gyroscopy patterns for Defenders — all of which are directly recoverable from wrist, chest, and ankle IMU data.

Three progressively expressive deep learning architectures are evaluated:
- 🔁 **M1** — LSTM + Temporal Self-Attention
- 🔄 **M2** — BiLSTM + Multi-Head Attention
- ⚡ **M3** — TCN + Transformer Encoder *(best: 99.24% accuracy)*

---

## ✨ Features

### 🗺️ Activity-to-Role Mapping (Core Contribution)
Sports-science-grounded mapping of 6 PAMAP2 activity classes to 3 football positional roles:

| PAMAP2 Activity | Football Role | IMU Basis |
|---|---|---|
| Running | Attacker | High ankle accel. peaks |
| Rope Jumping | Attacker | Vertical impulse transients |
| Walking | Midfielder | Sustained moderate accel. |
| Nordic Walking | Midfielder | Arm-swing stride coupling |
| Cycling | Defender | Rhythmic bilateral ankle cycle |
| Playing Soccer | Defender | Lateral reactive stops |

### 🧠 Three Deep Learning Architectures
- **M1 LSTM + Self-Attention** — additive temporal attention over 128-d hidden states
- **M2 BiLSTM + Multi-Head Attention** — 4-head scaled dot-product attention, 256-d bidirectional
- **M3 TCN + Transformer** — 4 dilated TemporalBlocks (receptive field 61 timesteps) → 2-layer Transformer encoder (GELU, dff=512)

### 🔍 SHAP Sensor Attribution
- 57-dimensional statistical window descriptor (mean, std, max per channel)
- RandomForest proxy + TreeExplainer on 100 test windows
- Per-role coaching-actionable sensor signatures identified

### ✅ LOSO Cross-Validation
- 5-fold StratifiedKFold approximating Leave-One-Subject-Out
- Generalises to unseen athletes: **98.89% ± 0.42%**
- Directly corrects the intra-subject leakage of the base paper

---

## 🖥️ Demo

### Role Prediction
```
Input  → 2s window | ankle accel peak: 18.3g | chest accel std: 0.4 | wrist gyro mean: 0.1
Output → ⚡ Attacker (P = 0.991)
         SHAP top signal: ankle_1_max, ankle_3_max
```

### SHAP Role Signatures
```
Attacker  → ankle_1_max, ankle_3_max    (explosive sprint/jump ground forces)
Midfielder → chest_2_std, chest_1_std   (directional diversity of box-to-box runs)
Defender  → hand_7_mean, hand_9_mean    (controlled lateral shuffle arm-swing)
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- GPU recommended (Kaggle T4/P100)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/bk1210/footballrole-dl.git
cd footballrole-dl
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download the Dataset
Get the PAMAP2 dataset from UCI:
👉 [PAMAP2 Physical Activity Monitoring Dataset](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)

Extract and place `.dat` files inside `PAMAP2_Dataset/Protocol/` in the project root.

### Step 4 — Run the Notebook
```bash
jupyter notebook FootballRole_DL.ipynb
```

Or upload directly to **Kaggle** and run with GPU.

---

## 📖 Usage

### Running the Full Pipeline

Open `FootballRole_DL.ipynb` and run all cells — the notebook handles:

1. PAMAP2 loading + preprocessing (2,724,953 rows, 55 columns)
2. Activity-to-role mapping (6 activities → 3 roles, ~1.1M frames)
3. 19-channel IoT feature selection (wrist/chest/ankle accel + gyro + HR)
4. 2-second sliding window segmentation (200 samples, 50% overlap, ~11,000 windows)
5. Train/val/test split (70/15/15, stratified)
6. Training M1, M2, M3 with cosine annealing + early stopping
7. LOSO 5-fold cross-validation on M1
8. SHAP TreeExplainer attribution per role class
9. Temporal self-attention weight visualisation

---

## 🏗️ Architecture

### Full Pipeline

```
PAMAP2 (2,724,953 rows | 9 subjects | 100 Hz)
    │
    ▼
19-channel IoT Selection
[HR | wrist accel/gyro | chest accel/gyro | ankle accel/gyro]
    │
    ▼
Activity-to-Role Mapping ← Core Contribution
[Running/RopeJump → Attacker | Walk/NordicWalk → Midfielder | Cycle/Soccer → Defender]
    │
    ▼
Sliding Window (T=200 samples, stride=100, 50% overlap)
→ ~11,000 labelled windows
    │
    ├──► M1: LSTM(128) × 2 → Self-Attention → FC(128→64→3)
    │
    ├──► M2: BiLSTM(128×2) × 2 → 4-Head MHA → FC
    │
    └──► M3: TCN (4 dilated blocks, δ=1,2,4,8) → Transformer(2L, 4-head) → GAP → FC
    │
    ▼
SHAP TreeExplainer (57-dim descriptor, RF proxy, 100 test windows)
→ Per-role sensor attribution maps
```

### Project Structure

```
footballrole-dl/
│
├── FootballRole_DL.ipynb        # Full pipeline — mapping, training, LOSO, SHAP
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 📊 Results

### Model Comparison vs Base Paper

| Model | Accuracy | Macro F1 | AUC | LOSO CV |
|---|---|---|---|---|
| Base Paper (CNN+BiLSTM) [Cuperman 2022] | 98.30% | 97.80% | 0.991 | ❌ None |
| M1: LSTM + Self-Attention | 98.81% | 98.66% | 0.999 | ✅ 98.89%±0.42% |
| M2: BiLSTM + Multi-Head Attn. | 99.07% | 98.91% | 0.999 | — |
| **M3: TCN + Transformer (best)** | **99.24%** | **99.11%** | **0.999** | — |

> ⚠️ Base paper evaluates on the **same subjects as training** — inflated accuracy with no cross-player generalisation. FootballRole-DL's LOSO confirms **98.89% on completely unseen athletes**.

### Per-Class Performance — M1

| Role | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Attacker | 0.991 | 0.988 | 0.989 | ~1,900 |
| Midfielder | 0.984 | 0.987 | 0.985 | ~2,050 |
| Defender | 0.987 | 0.984 | 0.985 | ~1,550 |
| **Macro Avg** | **0.987** | **0.986** | **0.987** | ~5,500 |

### SHAP Top Sensors Per Role

| Role | Top Sensor | |SHAP| | Biomechanical Meaning |
|---|---|---|---|
| Attacker | ankle_1_max, ankle_3_max | >0.18 | Explosive sprint/jump ground forces |
| Midfielder | chest_2_std, chest_1_std | — | Directional diversity of box-to-box runs |
| Defender | hand_7_mean, hand_9_mean | — | Lateral shuffle arm-swing patterns |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core language |
| PyTorch | M1/M2/M3 training, attention, TCN, Transformer |
| SHAP | TreeExplainer, per-role sensor attribution |
| scikit-learn | LOSO CV, RF proxy, metrics |
| SciPy | Majority vote window labeling |
| NumPy / Pandas | Data pipeline, feature extraction |
| Matplotlib / Seaborn | Confusion matrix, attention plots, SHAP bars |
| Kaggle GPU | Training environment |

---

## 📦 Dependencies

```txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
shap>=0.42.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

- [ ] Validate on STATSports / Catapult telemetry from professional training sessions
- [ ] Multi-label output for hybrid roles (e.g., wing-backs as Attacker + Defender)
- [ ] Graph Attention Network across player-IMU nodes for team-level tactical state recognition
- [ ] INT8 quantisation for on-device ARM Cortex-M edge inference in training vests
- [ ] Privacy-preserving federated learning across clubs

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Bharath Kesav R**
- 📧 Email: bharathkesav1275@gmail.com
- 🐙 GitHub: [@bk1210](https://github.com/bk1210)
- 🎓 Institution: Amrita Vishwa Vidyapeetham, Coimbatore

---

## 🙏 Acknowledgements

- [Reiss & Stricker (2012)](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) — PAMAP2 dataset
- [Cuperman & Jain (2022)](https://doi.org/10.3390/s22093484) — Base paper on wearable football activity recognition
- [Lundberg & Lee (2017)](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) — SHAP framework

---

<div align="center">

**⭐ If you found this project useful, please give it a star on GitHub! ⭐**

*Built with ❤️ for football/sports analytics and data analytics*

</div>
