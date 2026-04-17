# ============================================================
#  Heart Disease Prediction System
#  Based on M.Sc. Thesis – Niloofar Mastali
#  Ale-Taha Institute, Tehran – 2024
#  ML + Fuzzy Logic Approach
# ============================================================
# Run this in Google Colab or Jupyter Notebook
# Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

# ─────────────────────────────────────────────
# STEP 1: Install & Import Libraries
# ─────────────────────────────────────────────
# !pip install scikit-fuzzy  # uncomment if needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)

# All 8 algorithms from the thesis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

print("✅ Libraries imported successfully!")


# ─────────────────────────────────────────────
# STEP 2: Load & Explore Data
# ─────────────────────────────────────────────
# Option A: Load from file
# df = pd.read_csv('Heart Disease Dataset.csv')

# Option B: Load from URL (Kaggle mirror)
url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv"
try:
    df = pd.read_csv(url)
except:
    # Fallback: create sample data with correct structure
    print("⚠️  Could not load from URL. Creating sample dataset...")
    np.random.seed(42)
    n = 303
    df = pd.DataFrame({
        'age':       np.random.randint(29, 77, n),
        'sex':       np.random.randint(0, 2, n),
        'cp':        np.random.randint(0, 4, n),
        'trestbps':  np.random.randint(94, 200, n),
        'chol':      np.random.randint(126, 564, n),
        'fbs':       np.random.randint(0, 2, n),
        'restecg':   np.random.randint(0, 3, n),
        'thalach':   np.random.randint(71, 202, n),
        'exang':     np.random.randint(0, 2, n),
        'oldpeak':   np.round(np.random.uniform(0, 6.2, n), 1),
        'slope':     np.random.randint(0, 3, n),
        'ca':        np.random.randint(0, 4, n),
        'thal':      np.random.randint(0, 4, n),
        'target':    np.random.randint(0, 2, n),
    })

# Rename target if needed
if 'condition' in df.columns:
    df.rename(columns={'condition': 'target'}, inplace=True)

print(f"📊 Dataset shape: {df.shape}")
print(f"❤️  Disease cases: {df['target'].sum()} | Healthy: {(df['target']==0).sum()}")
print("\n", df.head())
print("\n📈 Statistics:")
print(df.describe().round(2))


# ─────────────────────────────────────────────
# STEP 3: Data Preprocessing
# ─────────────────────────────────────────────
print("\n🔍 Checking missing values...")
print(df.isnull().sum())

# Features & Target
features = ['age','sex','cp','trestbps','chol','fbs',
            'thalach','exang','oldpeak','ca']

X = df[features]
y = df['target']

# Split: 80% train, 10% validation, 10% test
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_remaining, y_remaining, test_size=0.5, random_state=42)

print(f"\n✂️  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

print("✅ Data preprocessing complete!")


# ─────────────────────────────────────────────
# STEP 4: Train All 8 Models
# ─────────────────────────────────────────────
models = {
    'Naïve Bayes':         GaussianNB(),
    'SVM':                 SVC(probability=True, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Neural Network':      MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42),
    'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
}

results = {}
print("\n🚀 Training all models...\n")

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        'model':     model,
        'accuracy':  acc,
        'precision': report['weighted avg']['precision'],
        'recall':    report['weighted avg']['recall'],
        'f1':        report['weighted avg']['f1-score'],
    }
    print(f"  {'✅' if acc >= 0.85 else '📊'} {name:<25} Accuracy: {acc:.2%}")

# Summary table
print("\n" + "="*65)
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("="*65)
for name, r in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    marker = " 🏆" if r['accuracy'] == max(v['accuracy'] for v in results.values()) else ""
    print(f"{name:<25} {r['accuracy']:>10.2%} {r['precision']:>10.2%} {r['recall']:>10.2%} {r['f1']:>10.2%}{marker}")
print("="*65)


# ─────────────────────────────────────────────
# STEP 5: Cross-Validation
# ─────────────────────────────────────────────
print("\n🔄 Running 5-Fold Cross-Validation...\n")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

X_full_sc = scaler.fit_transform(X)
cv_results = {}

for name, r in results.items():
    cv_scores = cross_val_score(r['model'], X_full_sc, y, cv=kf, scoring='accuracy')
    cv_results[name] = cv_scores
    print(f"  {name:<25} CV Mean: {cv_scores.mean():.2%} ± {cv_scores.std():.3f}")


# ─────────────────────────────────────────────
# STEP 6: Visualizations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#0a0e1a')
for ax in axes.flatten():
    ax.set_facecolor('#111827')

# --- Plot 1: Accuracy Comparison ---
ax1 = axes[0, 0]
sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'])
names  = [n for n, _ in sorted_models]
accs   = [r['accuracy'] for _, r in sorted_models]
colors = ['#ff4d6d' if a < 0.82 else '#ffd166' if a < 0.85 else '#00e5ff' for a in accs]
bars = ax1.barh(names, accs, color=colors, height=0.6, edgecolor='none')
for bar, acc in zip(bars, accs):
    ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{acc:.1%}', va='center', color='white', fontsize=10, fontweight='bold')
ax1.set_xlim(0.6, 1.02)
ax1.set_title('Model Accuracy Comparison', color='white', fontsize=13, fontweight='bold', pad=12)
ax1.tick_params(colors='#64748b')
ax1.spines[:].set_color('#1e2d45')
ax1.set_xlabel('Accuracy', color='#64748b')

# --- Plot 2: ROC Curves ---
ax2 = axes[0, 1]
roc_colors = ['#00e5ff','#b8ff57','#ffd166','#ff9a9e',
               '#64ffda','#80cbc4','#ff4d6d','#7ee8ff']
for i, (name, r) in enumerate(results.items()):
    if hasattr(r['model'], 'predict_proba'):
        y_prob = r['model'].predict_proba(X_test_sc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=roc_colors[i], lw=1.5,
                 label=f'{name} (AUC={roc_auc:.2f})', alpha=0.85)
ax2.plot([0,1],[0,1],'--', color='#64748b', lw=1, label='Random')
ax2.set_title('ROC Curves – All Models', color='white', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('False Positive Rate', color='#64748b')
ax2.set_ylabel('True Positive Rate', color='#64748b')
ax2.tick_params(colors='#64748b')
ax2.spines[:].set_color('#1e2d45')
ax2.legend(fontsize=7, labelcolor='white', facecolor='#161d2e',
           edgecolor='#1e2d45', loc='lower right')

# --- Plot 3: Confusion Matrix (Best Model) ---
ax3 = axes[1, 0]
best_name = max(results, key=lambda n: results[n]['accuracy'])
best_model = results[best_name]['model']
y_pred_best = best_model.predict(X_test_sc)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            linewidths=2, linecolor='#0a0e1a',
            annot_kws={'size':16, 'color':'white', 'weight':'bold'})
ax3.set_title(f'Confusion Matrix – {best_name}', color='white', fontsize=13,
              fontweight='bold', pad=12)
ax3.set_xlabel('Predicted', color='#64748b')
ax3.set_ylabel('Actual', color='#64748b')
ax3.tick_params(colors='#64748b')
ax3.set_xticklabels(['No Disease','Disease'], color='#64748b')
ax3.set_yticklabels(['No Disease','Disease'], color='#64748b', rotation=0)

# --- Plot 4: CV Score Distribution ---
ax4 = axes[1, 1]
cv_means = [cv_results[n].mean() for n in results.keys()]
cv_stds  = [cv_results[n].std()  for n in results.keys()]
x_pos = np.arange(len(results))
bars2 = ax4.bar(x_pos, cv_means, yerr=cv_stds, capsize=4,
                color='#00e5ff', alpha=0.7, edgecolor='none',
                error_kw={'color':'#ffd166','linewidth':2})
ax4.set_xticks(x_pos)
ax4.set_xticklabels([n.replace(' ','\n') for n in results.keys()],
                     fontsize=7, color='#64748b')
ax4.set_ylim(0.5, 1.05)
ax4.set_title('Cross-Validation Scores (5-Fold)', color='white', fontsize=13,
              fontweight='bold', pad=12)
ax4.set_ylabel('Accuracy', color='#64748b')
ax4.tick_params(colors='#64748b')
ax4.spines[:].set_color('#1e2d45')
ax4.axhline(y=0.85, color='#b8ff57', linestyle='--', lw=1, alpha=0.7, label='85% line')
ax4.legend(labelcolor='white', facecolor='#161d2e', edgecolor='#1e2d45', fontsize=9)

plt.suptitle('Heart Disease Prediction – Model Evaluation\nNiloofar Mastali · M.Sc. Thesis 2024',
             color='white', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('ml_results.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0e1a', edgecolor='none')
plt.show()
print("\n📊 Chart saved as ml_results.png")


# ─────────────────────────────────────────────
# STEP 7: Fuzzy Logic (Rule-Based)
# ─────────────────────────────────────────────
print("\n🔵 Applying Fuzzy Logic Risk Assessment...\n")

def fuzzy_heart_risk(age, chol, trestbps, thalach, oldpeak, ca, exang):
    """
    Simplified fuzzy logic system based on quartile membership.
    Inspired by Niloofar Mastali's thesis fuzzy approach in MATLAB.
    """
    score = 0.0

    # Age membership
    if   age < 40:              score += 0.1
    elif age < 55:              score += 0.3
    else:                       score += 0.6

    # Cholesterol membership
    if   chol < 200:            score += 0.1
    elif chol < 240:            score += 0.3
    else:                       score += 0.6

    # Blood pressure membership
    if   trestbps < 120:        score += 0.1
    elif trestbps < 140:        score += 0.3
    else:                       score += 0.6

    # Max heart rate (lower = higher risk)
    if   thalach > 160:         score += 0.1
    elif thalach > 130:         score += 0.3
    else:                       score += 0.6

    # ST depression
    if   oldpeak < 1.0:         score += 0.1
    elif oldpeak < 2.5:         score += 0.4
    else:                       score += 0.7

    # Vessels
    score += ca * 0.25

    # Exercise angina
    if exang == 1:              score += 0.5

    # Normalize to 0-1
    max_score = 0.6*5 + 0.75 + 0.5
    risk = score / max_score

    if   risk < 0.35:  return risk, "🟢 LOW RISK"
    elif risk < 0.65:  return risk, "🟡 MEDIUM RISK"
    else:              return risk, "🔴 HIGH RISK"


# Test on sample patients
print(f"{'Patient':<10} {'Age':>5} {'Chol':>6} {'BP':>5} {'HR':>5} {'OP':>5} {'CA':>4} {'EX':>4}   {'Result'}")
print("-"*75)
sample_patients = [
    ("Patient A", 45, 190, 115, 170, 0.5, 0, 0),
    ("Patient B", 62, 280, 155, 115, 3.2, 2, 1),
    ("Patient C", 55, 245, 138, 142, 1.5, 1, 0),
    ("Patient D", 70, 320, 170, 95,  4.0, 3, 1),
    ("Patient E", 38, 175, 110, 180, 0.2, 0, 0),
]
for p in sample_patients:
    name, age, chol, bp, hr, op, ca, ex = p
    risk_score, label = fuzzy_heart_risk(age, chol, bp, hr, op, ca, ex)
    print(f"{name:<10} {age:>5} {chol:>6} {bp:>5} {hr:>5} {op:>5} {ca:>4} {ex:>4}   {label} ({risk_score:.0%})")


# ─────────────────────────────────────────────
# STEP 8: Predict New Patient
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("🔮 PREDICT A NEW PATIENT")
print("="*55)

def predict_patient(age, sex, cp, trestbps, chol, fbs,
                    thalach, exang, oldpeak, ca):
    """Predict heart disease risk for a new patient."""
    patient = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp,
        'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'ca': ca
    }])
    patient_sc = scaler.transform(patient)

    print(f"\n👤 Patient: Age={age}, Sex={'M' if sex==1 else 'F'}, "
          f"BP={trestbps}, Chol={chol}, HR={thalach}")
    print("-"*50)

    probas = {}
    for name, r in results.items():
        pred = r['model'].predict(patient_sc)[0]
        if hasattr(r['model'], 'predict_proba'):
            prob = r['model'].predict_proba(patient_sc)[0][1]
            probas[name] = prob
        status = "❤️  DISEASE" if pred == 1 else "✅ HEALTHY"
        print(f"  {name:<25} → {status}")

    if probas:
        avg_prob = np.mean(list(probas.values()))
        print(f"\n  📊 Average probability of disease: {avg_prob:.1%}")
        if avg_prob > 0.65:
            print("  ⚠️  HIGH RISK — Please consult a physician!")
        elif avg_prob > 0.35:
            print("  ⚡ MODERATE RISK — Regular check-up recommended")
        else:
            print("  💚 LOW RISK — Keep up healthy habits!")

    # Fuzzy logic result
    fuzz_score, fuzz_label = fuzzy_heart_risk(age, chol, trestbps, thalach, oldpeak, ca, exang)
    print(f"\n  🔵 Fuzzy Logic Assessment: {fuzz_label}")


# Example prediction
predict_patient(
    age=58, sex=1, cp=2, trestbps=140, chol=268,
    fbs=0, thalach=130, exang=1, oldpeak=2.0, ca=2
)

print("\n" + "="*55)
print("✅ All done! Run predict_patient() with your own values.")
print("="*55)
print("""
FEATURES:
  age       - Age in years
  sex       - 1=Male, 0=Female
  cp        - Chest pain type (0-3)
  trestbps  - Resting blood pressure (mmHg)
  chol      - Cholesterol (mg/dl)
  fbs       - Fasting blood sugar >120 (1=True, 0=False)
  thalach   - Max heart rate achieved
  exang     - Exercise induced angina (1=Yes, 0=No)
  oldpeak   - ST depression (0-6)
  ca        - Number of major vessels (0-3)
""")
