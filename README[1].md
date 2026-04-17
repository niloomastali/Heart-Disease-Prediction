# ❤️ Heart Disease Prediction System
### ML + Fuzzy Logic Approach

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> **M.Sc. Thesis Project** · Niloofar Mastali · Ale-Taha Institute, Tehran · 2024

---

## 📌 Overview

This project presents a **hybrid heart disease prediction system** combining classical Machine Learning algorithms with Fuzzy Logic to improve diagnostic accuracy.

Eight classification algorithms were evaluated on the UCI Heart Disease Dataset, achieving up to **87% accuracy** with Naïve Bayes as the top performer.

---

## 🎯 Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Naïve Bayes** 🏆 | **87%** | **87%** | **87%** | **87%** |
| Support Vector Machine | 86% | 87% | 87% | 87% |
| Random Forest | 86% | 85% | 86% | 86% |
| K Nearest Neighbour | 85% | 86% | 86% | 85% |
| Logistic Regression | 85% | 85% | 85% | 85% |
| Neural Network | 84% | 84% | 84% | 84% |
| AdaBoost | 80% | 81% | 81% | 80% |
| Decision Tree | 75% | 77% | 76% | 75% |

---

## 🧠 Methodology

```
Dataset → Preprocessing → ML Models → Evaluation → Fuzzy Logic → Final Risk Score
```

**Two-stage approach:**
1. **Machine Learning** — 8 classification algorithms trained and compared
2. **Fuzzy Logic (MATLAB)** — Quartile-based membership functions for uncertainty handling

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── 📂 data/
│   └── heart_disease.csv          # UCI Heart Disease Dataset
│
├── 📂 notebooks/
│   └── heart_disease_ml.ipynb     # Main Jupyter Notebook
│
├── 📂 src/
│   ├── heart_disease_ml.py        # Full ML pipeline
│   └── fuzzy_logic.py             # Fuzzy logic implementation
│
├── 📂 results/
│   └── ml_results.png             # Charts and visualizations
│
├── README.md                      # This file
└── LICENSE                        # MIT License
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

Upload `heart_disease_ml.py` and run all cells.

### 4. Predict for a new patient
```python
predict_patient(
    age=58, sex=1, cp=2, trestbps=140,
    chol=268, fbs=0, thalach=130,
    exang=1, oldpeak=2.0, ca=2
)
```

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease Dataset
- **Samples:** 303 patients
- **Features:** 14 clinical attributes
- **Target:** Binary (0 = No Disease, 1 = Disease)

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | 1 = Male, 0 = Female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| ca | Number of major vessels (0–3) |

---

## 🔵 Fuzzy Logic

Beyond standard ML, this project applies **Fuzzy Logic in MATLAB** to handle medical uncertainty:

- Quartile-based membership functions per feature
- IF-THEN rule system for nuanced classification
- Output: **Low / Medium / High** risk categories

---

## 🛠️ Technologies

- **Python 3.8+** — Main programming language
- **Scikit-learn** — ML algorithms
- **Pandas / NumPy** — Data processing
- **Matplotlib / Seaborn** — Visualization
- **MATLAB** — Fuzzy Logic System
- **Google Colab** — Cloud execution

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is NOT a substitute for professional medical diagnosis. Always consult a qualified physician.

---

## 👩‍💻 Author

**Niloofar Mastali**
M.Sc. Computer Software Engineering (ML & AI)
Ale-Taha Institute · Tehran · 2024

📧 niloofar2020@gmail.com
🌍 Wien, Austria

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

*⭐ If you found this project helpful, please give it a star!*
