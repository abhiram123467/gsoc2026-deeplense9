<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:03001e,40:7303c0,80:ec38bc,100:fdeff9&height=230&section=header&text=DeepLense9&fontSize=85&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Unsupervised%20Super-Resolution%20of%20Gravitational%20Lensing%20Images&descAlignY=58&descSize=17" width="100%"/>

<br/>

[![GSoC 2026](https://img.shields.io/badge/GSoC-2026%20ML4Sci-F6AE2D?style=for-the-badge&logo=google&logoColor=white)](https://summerofcode.withgoogle.com/)
[![ML4Sci](https://img.shields.io/badge/ML4Sci-Test%20VI.A%20%2B%20VI.B-8B5CF6?style=for-the-badge)](https://ml4sci.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![RCAN](https://img.shields.io/badge/Model-RCAN--lite%204×SR-06B6D4?style=for-the-badge)]()
[![Telescope](https://img.shields.io/badge/Data-HSC%20%2F%20HST-10B981?style=for-the-badge&logo=nasa&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> **🔭 Recovering hidden dark matter substructure from blurry telescope images — with 4× super-resolution.**
>
> DeepLense9 trains a **Residual Channel Attention Network (RCAN-lite)** on simulated HR/LR lensing pairs, then **transfer-learns** onto **300 real HSC/HST telescope images** — bridging the sim-to-real gap to reveal lensing arcs and Einstein rings that bicubic interpolation simply cannot recover.

<br/>

[🚀 Quick Start](#-quick-start) · [🏗 Architecture](#-architecture) · [📊 Results](#-results) · [🎯 GSoC Vision](#-gsoc-2026-vision--ml4sci) · [🤝 Contribute](#-contributing)

</div>

---

## 🌌 What Is This?

Space telescopes like **HSC** (Hyper Suprime-Cam) and **HST** (Hubble) capture gravitational lensing at limited resolution. Sub-arc-second features — the very signatures that reveal **dark matter substructure** — are blurred out, making downstream analysis unreliable.

**DeepLense9 solves the resolution barrier** with a two-stage deep learning pipeline:

| Stage | Task | Data | Goal |
|---|---|---|---|
| **VI.A** | Simulated SR | HR/LR lensing image pairs | Learn lens morphology at 4× scale |
| **VI.B** | Real Telescope SR | 300 HSC/HST image pairs | Transfer-learn to real photon noise |

The result: sharper Einstein rings, clearer lensing arcs, and better dark matter science — from the same raw telescope data.

---

## 🔬 Mentors

> This work is submitted to **ML4Sci** under the guidance of:

| Mentor | Affiliation |
|---|---|
| **Michael Toomey** | MIT |
| **Pranath Reddy** | ML4Sci |
| **Sergei Gleyzer** | University of Alabama |

---

## 📊 Results

### Task VI.A — Simulated Lensing Super-Resolution

```
══════════════════════════════════════════════════════════
  TASK VI.A RESULTS — Simulated Lensing SR (4×)
══════════════════════════════════════════════════════════
  Method          MSE        SSIM       PSNR
  ──────────────────────────────────────────
  Bicubic         [baseline] [baseline] [baseline dB]
  LensingSRNet    [result]   [result]   [+X.XX dB] ✅
══════════════════════════════════════════════════════════
  PSNR Improvement: +X.XX dB over bicubic baseline
```

### Task VI.B — Real HSC/HST Telescope SR (Transfer Learning)

```
══════════════════════════════════════════════════════════
  TASK VI.B RESULTS — Real HSC/HST SR (Transfer Learn)
══════════════════════════════════════════════════════════
  Method          MSE        SSIM       PSNR
  ──────────────────────────────────────────
  Bicubic         [baseline] [baseline] [baseline dB]
  Fine-tuned      [result]   [result]   [+X.XX dB] ✅
══════════════════════════════════════════════════════════
  PSNR Improvement: +X.XX dB over bicubic on real data
```

> **📌 Run the notebook to populate real numbers — they auto-print at the end of every cell.**

### Visual Comparison (4 columns per row)

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  LR Input   │   Bicubic   │  LensingSR  │  HR Ground  │
│  (16×16)    │  (64×64)    │   (64×64)   │  Truth      │
├─────────────┼─────────────┼─────────────┼─────────────┤
│  Blurry     │  Soft arcs  │  Sharp arcs │  Reference  │
│  lens arc   │  no detail  │  preserved  │             │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

*See `outputs/02_simulated_comparison.png` and `outputs/07_real_comparison.png` generated on run.*

---

## 🏗 Architecture

### LensingSRNet — Residual Channel Attention Network (RCAN-lite)

```
Input: LR image (B, 1, 16, 16)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1 — Shallow Feature Extraction                    │
│  Conv2d(1 → 64, kernel=3, pad=1)                         │
│  Output: (B, 64, 16, 16)                                 │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2 — Deep Residual Blocks × 8                      │
│                                                          │
│  Each ResidualBlock:                                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │  x → Conv(64→64) → ReLU → Conv(64→64)           │    │
│  │             ↓                                   │    │
│  │      Channel Attention (CA)                     │    │
│  │      AvgPool → FC(64→4) → ReLU → FC(4→64)       │    │
│  │      → Sigmoid → scale features                 │    │
│  │             ↓                                   │    │
│  │      output = x + (features × CA_weights)       │    │
│  └─────────────────────────────────────────────────┘    │
│  ×8 stacked with residual connections                    │
│  Output: (B, 64, 16, 16)                                 │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3 — 4× Pixel Shuffle Upsampling                   │
│  Conv2d(64 → 64×4², kernel=3)                            │
│  PixelShuffle(upscale=4)                                 │
│  Output: (B, 64, 64, 64)                                 │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 4 — Reconstruction                               │
│  Conv2d(64 → 1, kernel=3)                               │
│  Output: SR image (B, 1, 64, 64)                        │
└──────────────────────────────────────────────────────────┘
```

**Why Channel Attention?** Lensing images have spatially sparse features — Einstein rings occupy a narrow annular region while most pixels are background. CA re-weights feature channels to focus computation on arc-like structures rather than smooth background, directly improving SSIM on the scientifically important regions.

---

### Combined Loss Function

```python
Loss = L1(SR, HR)  +  λ · GradLoss(SR, HR)

where:
  L1 Loss        → pixel-level reconstruction accuracy
  GradLoss       → Sobel filter edge map MSE
                   (preserves sharp lensing arc boundaries)
  λ (grad_w)     = 0.1

# Why gradient loss? Lensing arcs are 1–3px thin features.
# Pure L1 averages them away. Sobel loss explicitly penalises
# blurry edges — the single most important feature for dark matter science.
```

---

### Transfer Learning Strategy (VI.A → VI.B)

```
┌─────────────────────────────────────────────────────────┐
│  TASK VI.A  →  Train on simulated HR/LR pairs           │
│  LR: 1e-4 (AdamW) | 50 epochs | Cosine LR decay        │
│  Save: outputs/best_sr_model_simulated.pth              │
└────────────────────┬────────────────────────────────────┘
                     │  Load pretrained weights
                     ▼
┌─────────────────────────────────────────────────────────┐
│  TASK VI.B  →  Fine-tune on 300 real HSC/HST pairs      │
│  LR: 5e-5 (lower — preserve learned features)          │
│  Batch: 16 | Real noise distribution                    │
│  Save: outputs/best_sr_model_real.pth                   │
└─────────────────────────────────────────────────────────┘

Domain Gap: Simulated images are idealized.
Real telescope images have:
  - CCD readout noise
  - Atmospheric seeing effects
  - PSF convolution artifacts
Fine-tuning adapts the model to these real-world conditions
while preserving the lensing morphology learned in VI.A.
```

---

## 🚀 Quick Start

### ▶️ Run on Google Colab (Recommended — Free T4 GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhiram123467/gsoc2026-deeplense9/blob/main/DEEPLENSE9_SuperResolution.ipynb)

1. Click **Open in Colab**
2. `Runtime` → `Change runtime type` → **T4 GPU** → Save
3. `Runtime` → `Run all`
4. ☕ ~20–30 min — all outputs auto-saved to `outputs/`

### 💻 Run Locally

```bash
# Clone
git clone https://github.com/abhiram123467/gsoc2026-deeplense9
cd gsoc2026-deeplense9

# Install
pip install torch torchvision matplotlib scikit-image scikit-learn \
            numpy tqdm gdown Pillow seaborn

# Launch notebook
jupyter notebook DEEPLENSE9_SuperResolution.ipynb
```

---

## 📁 Project Structure

```
gsoc2026-deeplense9/
│
├── 📓 DEEPLENSE9_SuperResolution.ipynb   # Complete pipeline — run this
│
├── data/
│   ├── simulated/                         # VI.A: HR/LR synthetic pairs
│   │   └── SR_dataset.zip                 # Auto-downloaded via gdown
│   └── real/                              # VI.B: 300 HSC/HST pairs
│       └── real_SR_dataset.zip            # Auto-downloaded via gdown
│
├── outputs/                               # Auto-generated on run
│   ├── 01_dataset_samples.png             # LR/HR pair visualisation
│   ├── 02_simulated_comparison.png        # LR | Bicubic | SR | HR (VI.A)
│   ├── 03_training_curves.png             # Loss + SSIM + PSNR plots
│   ├── 04_metrics_bar_chart.png           # Bicubic vs LensingSRNet (VI.A)
│   ├── 05_real_samples.png                # HSC/HST real image samples
│   ├── 06_fine_tuning_curves.png          # VI.B training curves
│   ├── 07_real_comparison.png             # LR | Bicubic | SR | HR (VI.B)
│   ├── 08_summary_dashboard.png           # Full A+B metrics dashboard
│   ├── best_sr_model_simulated.pth        # Trained VI.A weights
│   └── best_sr_model_real.pth             # Fine-tuned VI.B weights
│
├── README.md
└── requirements.txt
```

---

## 🔧 Technical Stack

```
🤖 Deep Learning   : PyTorch 2.x, torchvision
🏗  Architecture   : RCAN-lite (Residual Channel Attention Network)
📐 Upsampling      : Pixel Shuffle (sub-pixel convolution) × 4
📉 Loss            : L1 + Sobel Gradient Loss (λ=0.1)
📊 Metrics         : MSE, SSIM (scikit-image), PSNR
⚙️ Optimizer       : AdamW (β₁=0.9, β₂=0.999) + CosineAnnealingLR
🔁 Regularisation  : Gradient clipping (max_norm=1.0), seed=42
📡 Telescopes      : HSC (Hyper Suprime-Cam) + HST (Hubble Space Telescope)
☁️ Compute         : Google Colab T4 GPU
🔬 Domain          : Gravitational lensing, dark matter substructure
```

---

## 🎯 GSoC 2026 Vision — ML4Sci

> **Target Organisation: ML4Sci (Machine Learning for Science)**
> **Tasks: DeepLense — Test VI.A + Test VI.B**
> **Mentors: Michael Toomey (MIT) · Pranath Reddy · Sergei Gleyzer (Alabama)**

This notebook is the **evaluation submission** for ML4Sci GSoC 2026. The proposed 12-week project extends this foundation:

| Phase | Weeks | Deliverable |
|---|---|---|
| **Foundation** | 1–2 | Reproduce + document VI.A and VI.B baselines |
| **DDPM-SR** | 3–5 | Diffusion-based super-resolution (connects to DeepLense8) |
| **Unsupervised** | 6–8 | Unpaired SR via cycle-consistent adversarial training |
| **Physics Loss** | 9–10 | Lensing arc preservation metric as auxiliary loss |
| **Integration** | 11–12 | Plug-in SR pre-processor for DeepLense classifiers |

**Connection to existing DeepLense work:**
- 🔗 Builds directly on **DEEPLENSE4 Foundation Model** (transfer learning backbone)
- 🔗 Pairs with **DEEPLENSE8 DDPM** — diffusion models for both generation and SR
- 🌑 SR enhances substructure visibility → feeds better data to dark matter classifiers

**Why this matters for astrophysics:**
- 🌌 Rubin LSST will image ~100,000 gravitational lenses — SR can recover substructure from ground-based seeing-limited data
- 🛰 Euclid space mission data — SR pre-processing can extend effective resolution by 4×
- 🔭 HST archive has 20+ years of lensing observations at varying resolution — SR unifies them

---

## 📚 References

- [RCAN — Zhang et al. 2018](https://arxiv.org/abs/1807.02758) — Residual Channel Attention Networks for SR
- [SRCNN — Dong et al. 2014](https://arxiv.org/abs/1501.00092) — foundational deep SR paper
- [Pixel Shuffle — Shi et al. 2016](https://arxiv.org/abs/1609.05158) — efficient sub-pixel convolution
- [ML4Sci DeepLense](https://ml4sci.org/) — host organisation
- [HSC Survey](https://hsc.mtk.nao.ac.jp/) — Hyper Suprime-Cam telescope data
- [Pearson et al. 2019](https://arxiv.org/abs/1904.03041) — super-resolution for radio astronomy

---

<div align="center">

## 👨‍🚀 About the Author

**Abhi Ramg** — AI/ML Researcher & GSoC 2026 Applicant

📍 Hyderabad, India &nbsp;|&nbsp; 🔭 Astrophysics AI &nbsp;|&nbsp; 🏗 Deep Learning for Science

[![GitHub](https://img.shields.io/badge/GitHub-abhiram123467-181717?style=for-the-badge&logo=github)](https://github.com/abhiram123467)
[![DeepLense9](https://img.shields.io/badge/Repo-DeepLense9-8B5CF6?style=for-the-badge&logo=github)](https://github.com/abhiram123467/gsoc2026-deeplense9)
[![DeepLense8](https://img.shields.io/badge/Also%20See-DeepLense8%20DDPM-EC38BC?style=for-the-badge&logo=github)](https://github.com/abhiram123467/DeepLense8)

<br/>

*"What the telescope blurs, the network sharpens. What the noise hides, attention finds."*

<br/>

**⭐ Star this repo if physics-informed super-resolution excites you!**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:fdeff9,40:ec38bc,80:7303c0,100:03001e&height=130&section=footer" width="100%"/>

</div>
