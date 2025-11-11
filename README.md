# **AN AUDIO DEEPFAKE DETECTION METHOD BASED ON THE LARGE-SCALE MULTILINGUAL SELF-SUPERVISED TRANSFORMER MODEL XLS-R**

## 🔍 Overview

This repository presents an **audio deepfake detection** system leveraging **XLS-R**, a large-scale multilingual self-supervised Transformer model.
The approach focuses on detecting **spoofed or synthetic speech** in the **ASVspoof 2019 Logical Access (LA)** dataset, which contains real and artificially generated audio samples.

The model was **trained and evaluated on Kaggle** using a **NVIDIA Tesla P100 GPU** environment.

---

## 🧠 Model Description

* **Backbone Model:** [XLS-R (Cross-Lingual Speech Representation Model)](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
* **Framework:** PyTorch & Hugging Face Transformers
* **Fine-tuning Strategy:**

  * Stage 1: Train classification head while freezing XLS-R encoder layers.
  * Stage 2: Fine-tune the top Transformer layers for optimal representation learning.
* **Task:** Binary classification — detecting **real vs. fake** speech.

---

## 🧾 Dataset

* **Dataset:** [ASVspoof 2019 Logical Access (LA)](https://datashare.ed.ac.uk/handle/10283/3336)
* **Sampling Rate:** 16 kHz
* **Preprocessing:**

  * Audio resampled to 16 kHz using `torchaudio`.
  * 5-second waveform segments standardized via `Wav2Vec2FeatureExtractor`.

---

## ⚙️ Training Environment

| Item       | Specification                      |
| ---------- | ---------------------------------- |
| Platform   | Kaggle                             |
| GPU        | NVIDIA Tesla P100                  |
| Frameworks | PyTorch, Transformers              |
| Precision  | FP32                               |
| Epochs     | Variable (based on validation EER) |
| Logging    | Manual (print-based tracking)      |

---

## 📈 Performance Summary

| Dataset                         | EER (Equal Error Rate) | Notes                        |
| ------------------------------- | ---------------------- | ---------------------------- |
| ASVspoof 2019 LA (Dev)          | **1.06%**              | After fine-tuning            |
| Baseline XLS-R (frozen encoder) | 2.28%                  | Before fine-tuning           |
| Traditional Baseline System     | ≈ 8%                   | From ASVspoof 2019 challenge |
| Top Challenge System            | ≈ 0.22%                | Complex ensemble             |

> Even without ensemble techniques, our single XLS-R model achieves near state-of-the-art performance on ASVspoof 2019 LA.

---

## 📂 Repository Structure

```
├── wav2vec2-xls-r-base-split.ipynb
├── requirements.txt                      # Python dependencies
└── README.md                             # Project description (this file)
```

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Create environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the exported script
python src/wav2vec2_xls_r_base_split.py
```

---

## 📚 Dependencies

Main packages required:

```
torch
torchaudio
transformers
tqdm
pandas
numpy
matplotlib
scikit-learn
```

---

## 📄 License

This project is released under the **MIT License**.
You may freely use, modify, and distribute it for research or educational purposes.


