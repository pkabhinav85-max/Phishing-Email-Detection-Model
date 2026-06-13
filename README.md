# 🤖 AI-Powered Phishing Email Detector

A machine learning classification framework built using Python and `scikit-learn`. This security utility leverages **Natural Language Processing (NLP)** and supervised learning to statically analyze email text payloads, accurately classifying them as **"Phishing"** or **"Safe"** based on behavioral indicators, urgency keywords, and semantic URL structures.

Developed as a core technical milestone during my **Cyber Security Internship** at **Thiranex**.

---

## 🚀 Key Features

- **NLP Text Vectorization:** Implements `TfidfVectorizer` to convert raw strings into numerical matrices, heavily weighting high-signal fraud terms (*urgent, suspended, verify, click*) while ignoring benign stop words.
- **Supervised ML Classification:** Utilizes a calibrated `Logistic Regression` classifier to compute statistical weights and separate legitimate communications from social engineering attacks along a sigmoid curve.
- **Detailed Security Diagnostics:** Automatically evaluates the model's performance on a testing split, displaying accuracy scores, a comprehensive classification report, and a confusion matrix.
- **Interactive Testing Sandbox:** Features a real-time terminal command loop allowing security analysts to paste custom email strings and view an instant classification verdict complete with a confidence percentage.

---

## ⚙️ Core Cybersecurity & Data Science Concepts Applied

1. **Feature Engineering (TF-IDF):** Understanding how text is mathematically parsed to highlight critical indicators of compromise (IoCs) and manipulation phrases.
2. **Binary Classification Modeling:** Training an algorithmic model to make deterministic, high-confidence security decisions (Is this threat group `0` or `1`?).
3. **Confusion Matrix Auditing:** Dissecting model mistakes. In a real-world enterprise environment, minimizing **False Negatives** (missing a malicious email that lands in a user's inbox) takes priority over minimizing **False Positives** (accidentally filtering a clean message to spam).

---

## 🛠️ Step-by-Step Local Deployment

### 1. Install Dependencies
Ensure you have the required Python libraries installed via your system package manager (recommended for Kali Linux to comply with PEP 668):
```bash
sudo apt update && sudo apt install -y python3-sklearn python3-pandas python3-numpy
