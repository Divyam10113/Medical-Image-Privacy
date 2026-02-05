# Privacy-Preserving Medical Image Synthesis
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A research implementation of **Denoising Diffusion Probabilistic Models (DDPM)** for generating synthetic medical images (Chest X-rays) while evaluating privacy risks.

## 🏥 Practical Use Case
**Problem:** Medical institutions cannot share patient data due to strict privacy regulations (HIPAA/GDPR). This hinders the progress of AI in healthcare.

**Solution:** Train a generative model on private data to produce **synthetic** images. These synthetic images follow the same statistical distribution as real data (and can be used to train downstream classifiers) but do not map to any specific real individual.

**Research Contribution:**
1.  **Privacy Evaluation:** We simulate a **Membership Inference Attack (MIA)** to strictly quantify if the model has "memorized" any specific patient.
2.  **Quality Metrics:** We use **FID (Fréchet Inception Distance)** to ensure the synthetic X-rays are medically realistic.

## 🏗️ Architecture
*   **Model:** U-Net with DDPM Scheduler (1000 timesteps).
*   **Dataset:** NIH Chest X-ray Dataset.
*   **Attacks:** Reconstruction Loss-based Membership Inference.

## 🚀 Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training
Train the diffusion model on your local machine or cloud GPU.
```bash
python src/train.py --epochs 50 --batch_size 32
```

### 3. Generation
Generate synthetic X-rays and calculate FID score.
```bash
python src/generate.py
```

### 4. Privacy Audit
Run the Membership Inference Attack to test for overfitting/memorization.
```bash
python run_attack.py
```

## 📊 Results
*   **FID Score:** ~1.41 (High Realism)
*   **Privacy Assessment:** Model showed susceptibility to reconstruction attacks (Future work: Implement Differential Privacy).

## 📂 Project Structure
```
├── data/               # Dataset storage
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # U-Net and Diffusion logic
│   ├── eval/           # Metrics (FID) and Privacy Attacks
│   ├── train.py        # Main training loop
│   └── generate.py     # Inference script
└── requirements.txt
```
