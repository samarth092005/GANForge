# GANForge: Custom DCGAN Fine-Tuning with AI Auto-Report

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![Groq](https://img.shields.io/badge/Groq-API-green.svg)
![HuggingFace](https://img.shields.io/badge/BLIP-Image%20Captioning-yellow.svg)

## 1. Academic Problem Statement
**P40 – Custom DCGAN Fine-Tune with Auto-Report**
* **Goal**: Fine-tune a Deep Convolutional GAN (DCGAN) on a custom image dataset (~300 images, e.g., Indian Rangoli art) and automatically generate an end-to-end technical training report.
* **Mandatory Units**:
  * **U1**: Collect, clean, preprocess (64x64, [-1, 1] normalization), and split dataset.
  * **U2**: Train/Fine-tune DCGAN; LLM writes the report directly from genuine training logs and loss curves.
  * **U3**: Describe generated sample grids across epochs using BLIP (`Salesforce/blip-image-captioning-base`).
  * **U4**: Track Fréchet Inception Distance (FID) per checkpoint epoch using TensorFlow 2 InceptionV3 embeddings.
  * **U5**: Record dataset licensing, image consent provenance, memorization analysis, failure cases, and Responsible AI guidelines.

---

## 2. Architecture & Directory Structure

```
GANForge/
├── app.py                      # Main Streamlit Dashboard Application
├── requirements.txt            # Dependency specification
├── .env.example                # Template for environment secrets
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
│
├── config/
│   ├── __init__.py
│   └── settings.py             # Hyperparameters, paths, and configurations
│
├── data/
│   ├── raw/                    # Raw uploaded/collected images
│   ├── processed/              # Normalized, validated, preprocessed images
│   └── dataset_loader.py       # (Phase 2) Cleaning, resizing, splitting
│
├── models/
│   ├── __init__.py
│   ├── generator.py            # (Phase 3) DCGAN Generator
│   ├── discriminator.py        # (Phase 3) DCGAN Discriminator
│   └── dcgan.py                # (Phase 3) Combined Adversarial Model
│
├── training/
│   ├── __init__.py
│   ├── train.py                # (Phase 4) Custom training loop & losses
│   ├── callbacks.py            # (Phase 4) Logging & sample generation
│   └── checkpoint.py           # (Phase 4) Model saving & restore
│
├── evaluation/
│   ├── __init__.py
│   ├── fid_score.py            # (Phase 5) InceptionV3 FID computation
│   └── metrics.py              # (Phase 5) Metric history & charting
│
├── services/
│   ├── __init__.py
│   ├── groq_service.py         # (Phase 7) Groq LLM integration
│   └── blip_service.py         # (Phase 6) BLIP Image Captioning
│
├── reports/
│   ├── __init__.py
│   └── report_generator.py     # Structured Auto-Report compilation
│
├── outputs/
│   ├── samples/                # Generated grids at milestones
│   ├── checkpoints/            # Generator/Discriminator weights
│   └── metrics/                # Loss and FID histories (.json / .csv)
│
└── notebooks/
    └── dcgan_training.ipynb    # Google Colab GPU training notebook
```

---

## 3. Local Setup Instructions

### Prerequisites
* **OS**: Windows / Linux / macOS
* **Python**: 3.12.x
* **Terminal**: PowerShell / Bash

### Quick Start
1. **Activate Virtual Environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Configure Environment Secrets**:
   ```powershell
   Copy-Item .env.example .env
   # Edit .env and insert your GROQ_API_KEY
   ```
4. **Launch Dashboard** *(available after Phase 8)*:
   ```powershell
   streamlit run app.py
   ```
