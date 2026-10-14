# 🛡️ Data Exfiltration Detection on DNS-over-HTTPS (DoH) Using Flow Metadata Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doh-exfiltration-detection.streamlit.app)
[![Dataset](https://img.shields.io/badge/Dataset-BCCC--CIRA--CIC--DoHBrw--2020-green.svg)](https://www.unb.ca/cic/datasets/dohbrw-2020.html)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.98%25-brightgreen.svg)]()
[![Privacy](https://img.shields.io/badge/Privacy-Non--Payload%20Approach-orange.svg)]()

> **Undergraduate Thesis (Skripsi) Research — Universitas Widyatama (2026)**  
> **Author:** Yogi Mugni Santoso  
> **Topic:** Detection of DNS-over-HTTPS (DoH) Data Exfiltration and Tunneling Attacks Using Non-Payload Flow Metadata Machine Learning.  
> **Live Demo:** [doh-exfiltration-detection.streamlit.app](https://doh-exfiltration-detection.streamlit.app)

---

## 📌 Overview

**DNS-over-HTTPS (DoH — RFC 8484)** encrypts DNS queries inside HTTPS port 443, securing user privacy from eavesdropping. However, adversaries exploit DoH as a covert communication channel for data exfiltration (via Base64/Hex encoded subdomains) and Command-and-Control (C2) tunneling.

Traditional Intrusion Detection Systems (IDS) relying on Deep Packet Inspection (DPI) cannot inspect TLS-encrypted payloads without privacy-invasive decryption. This research proposes a **privacy-preserving, non-payload machine learning architecture** using only statistical flow metadata.

---

## 🚀 Key Research Findings

| Metric | Logistic Regression | AdaBoost | Random Forest (Best) |
|---|---|---|---|
| **Accuracy** | 90.13% | 99.82% | **99.98%** |
| **Recall** | 94.23% | 99.79% | **99.97%** |
| **F1-Score** | 90.64% | 99.82% | **99.98%** |
| **False Positive Rate (FPR)** | 0.140810 | 0.001444 | **0.000041** |
| **Inference Latency** | 0.0011 ms | 0.0120 ms | **0.0036 ms/sample** |

- **Optimal Operating Threshold ($\theta = 0.45$):** Reduces False Negatives from 11 to 9 instances with only 2 False Positives out of 48,491 benign samples.
- **Explainability (SHAP & Gini):** `PacketLengthMode` (Gini: 0.2251) is the dominant discriminator due to malware's fixed-size Base64 chunking mechanism.

---

## 🛠️ Project Structure

```text
├── app.py                     # Interactive Streamlit GUI (Dashboard & Prediction)
├── Poster.html                # Print-Ready A4 Portrait Academic Research Poster
├── requirements.txt           # Python dependencies for deployment
├── requirements_gui.txt       # GUI dependencies
├── lampiran_2_source_code.py  # Model training and feature evaluation scripts
├── .gitignore                 # Excludes venv, cache, and heavy dataset (>100MB)
├── notebooks/                 # Exploratory data analysis notebooks
└── outputs/                   # Trained models (.pkl), evaluation metrics, and plots
```

---

## 💻 Installation & Usage

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/sskntlmns/doh_exfiltration_skripsi.git
cd doh_exfiltration_skripsi
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit GUI Dashboard
```bash
streamlit run app.py
```

### 4. View / Print Research Poster
Open `Poster.html` in any modern web browser and press `Ctrl + P` (Layout: Portrait, enable "Background graphics").

---

## 📜 Citation & Acknowledgments
- Dataset: **BCCC-CIRA-CIC-DoHBrw-2020** (Canadian Institute for Cybersecurity).
- Thesis Supervisor & Informatics Engineering Department, **Universitas Widyatama, Bandung**.
