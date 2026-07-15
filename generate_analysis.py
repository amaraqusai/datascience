"""
Comprehensive Phishing Website Detection Analysis
===================================================
Final Project - Data Science in Cybersecurity

Source: sujeetgund/phishing-website-detection
Dataset: UCI Phishing Websites (11,055 rows, 30 features)

This script performs ALL required empirical analyses and saves:
  - All figures to images/
  - All metrics to metrics.json
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import (RFE, mutual_info_classif,
                                       SelectKBest)
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, matthews_corrcoef,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve)

np.random.seed(42)
os.makedirs('images', exist_ok=True)

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 60)
print("SECTION 1: DATA LOADING")
print("=" * 60)

df = pd.read_csv('data/phishingData.csv')
df.columns = df.columns.str.lower()

print(f"Shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nData types:\n{df.dtypes.value_counts()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ============================================================
# 2. THOROUGH EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 2.1 Missing values
print("\n--- 2.1 Missing Values ---")
missing = df.isnull().sum()
print(f"Total missing values: {missing.sum()}")
print(f"Columns with missing values: {list(missing[missing > 0].index)}")

# 2.2 Infinite values
print("\n--- 2.2 Infinite Values ---")
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print(f"Total infinite values: {inf_count}")

# 2.3 Constant features (nunique <= 1)
print("\n--- 2.3 Constant Features ---")
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
print(f"Constant features: {constant_cols if constant_cols else 'None'}")

# 2.4 Single-value-dominant features (>95% one value)
print("\n--- 2.4 Single-Value-Dominant Features ---")
for col in df.columns:
    top_pct = df[col].value_counts(normalize=True).iloc[0]
    if top_pct > 0.95:
        print(f"  {col}: {top_pct:.1%} is value {df[col].value_counts().index[0]}")

# 2.5 Unique values per column
print("\n--- 2.5 Unique Values Per Column ---")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique -> {sorted(df[col].unique())}")

# 2.6 Exact duplicate rows
print("\n--- 2.6 Exact Duplicate Rows ---")
dup_count = df.duplicated().sum()
print(f"Exact duplicate rows: {dup_count} ({dup_count/len(df)*100:.2f}%)")

# 2.7 Duplicate feature vectors (same X, different y)
print("\n--- 2.7 Conflicting Labels (same features, different result) ---")
X_cols = [c for c in df.columns if c != 'result']
dup_features = df[df.duplicated(subset=X_cols, keep=False)]
conflicting = dup_features.groupby(X_cols)['result'].nunique()
conflicting_count = (conflicting > 1).sum()
print(f"Feature vectors with conflicting labels: {conflicting_count}")

# 2.8 Class distribution
print("\n--- 2.8 Class Distribution ---")
class_dist = df['result'].value_counts()
print(f"Class counts:\n{class_dist}")
imbalance_ratio = class_dist.max() / class_dist.min()
print(f"Imbalance ratio: {imbalance_ratio:.2f}")

fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(['Legitimate (-1)', 'Phishing (1)'],
              [class_dist.get(-1, 0), class_dist.get(1, 0)],
              color=colors, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, [class_dist.get(-1, 0), class_dist.get(1, 0)]):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
            f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Class Distribution: Legitimate vs Phishing')
ax.set_ylim(0, max(class_dist) * 1.15)
plt.tight_layout()
plt.savefig('images/01_class_distribution.png', dpi=150)
plt.close()

# 2.9 Feature distributions
print("\n--- 2.9 Feature Distributions ---")
fig, axes = plt.subplots(6, 5, figsize=(20, 20))
axes = axes.flatten()
feature_cols = [c for c in df.columns if c != 'result']
for i, col in enumerate(feature_cols):
    ax = axes[i]
    for label, color in [(-1, '#2ecc71'), (1, '#e74c3c')]:
        subset = df[df['result'] == label][col]
        ax.hist(subset, bins=5, alpha=0.6, color=color, label=f'{"Legit" if label==-1 else "Phish"}', edgecolor='black')
    ax.set_title(col, fontsize=8)
    ax.tick_params(labelsize=6)
    if i == 0:
        ax.legend(fontsize=6)
plt.suptitle('Feature Distributions by Class', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('images/02_feature_distributions.png', dpi=150)
plt.close()

# 2.10 Outlier analysis (IQR method)
print("\n--- 2.10 Outlier Analysis (IQR) ---")
outlier_summary = {}
for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    if outliers > 0:
        outlier_summary[col] = outliers
        print(f"  {col}: {outliers} outliers ({outliers/len(df)*100:.2f}%)")
if not outlier_summary:
    print("  No outliers detected (features are ordinal -1, 0, 1)")

# 2.11 Correlation analysis (Spearman, Pearson, Kendall)
print("\n--- 2.11 Correlation Analysis ---")

# Spearman
fig, ax = plt.subplots(figsize=(16, 14))
corr_spearman = df.corr(method='spearman')
mask = np.triu(np.ones_like(corr_spearman, dtype=bool))
sns.heatmap(corr_spearman, mask=mask, annot=False, cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/03_correlation_spearman.png', dpi=150)
plt.close()

# Top correlations with target
target_corr = corr_spearman['result'].drop('result').abs().sort_values(ascending=False)
print(f"\nTop 10 features correlated with target (Spearman |r|):")
for feat, val in target_corr.head(10).items():
    direction = "+" if corr_spearman.loc[feat, 'result'] > 0 else "-"
    print(f"  {feat}: {direction}{val:.4f}")

# 2.12 Redundancy analysis (highly correlated feature pairs)
print("\n--- 2.12 Redundancy Analysis ---")
high_corr_pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        c = abs(corr_spearman.loc[feature_cols[i], feature_cols[j]])
        if c > 0.5:
            high_corr_pairs.append((feature_cols[i], feature_cols[j], c))
high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
print(f"Feature pairs with |Spearman r| > 0.5:")
for f1, f2, c in high_corr_pairs:
    print(f"  {f1} <-> {f2}: {c:.4f}")

# 2.13 Group-by analysis (mean feature values by class)
print("\n--- 2.13 Group-By Analysis (Mean by Class) ---")
group_means = df.groupby('result')[feature_cols].mean()
diff = (group_means.loc[1] - group_means.loc[-1]).abs().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 6))
x = range(len(diff))
ax.bar(x, diff.values, color='steelblue', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(diff.index, rotation=90, fontsize=7)
ax.set_ylabel('|Mean(Phishing) - Mean(Legitimate)|')
ax.set_title('Absolute Difference in Feature Means Between Classes', fontweight='bold')
plt.tight_layout()
plt.savefig('images/04_groupby_mean_diff.png', dpi=150)
plt.close()

print(f"\nTop 10 features with largest mean difference between classes:")
for feat, val in diff.head(10).items():
    print(f"  {feat}: {val:.4f}")

# 2.14 Crosstab analysis
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
top_features = diff.head(6).index.tolist()
for idx, feat in enumerate(top_features):
    ax = axes[idx // 3][idx % 3]
    ct = pd.crosstab(df[feat], df['result'], normalize='index')
    ct.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c'],
            edgecolor='black', linewidth=0.5)
    ax.set_title(f'{feat} vs Result', fontsize=10)
    ax.set_ylabel('Proportion')
    ax.legend(['Legit (-1)', 'Phish (1)'], fontsize=7)
    ax.tick_params(labelsize=8)
plt.suptitle('Top 6 Discriminative Features: Crosstab Analysis', fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('images/05_crosstab_top_features.png', dpi=150)
plt.close()

# Drop duplicates for modeling
df = df.drop_duplicates()
print(f"\nShape after dropping {dup_count} duplicates: {df.shape}")

# ============================================================
# 3. DATA PREPARATION (Leakage-safe splitting)
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: DATA PREPARATION")
print("=" * 60)

X = df.drop(columns=['result'])
y = df['result'].map({-1: 0, 1: 1})

# Stratified 60/20/20 split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)  # 0.25 of 0.80 = 0.20

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train class dist: {y_train.value_counts().to_dict()}")
print(f"Val class dist:   {y_val.value_counts().to_dict()}")
print(f"Test class dist:  {y_test.value_counts().to_dict()}")

# 2.15 Leakage check
print("\n--- Leakage Check ---")
train_idx = set(X_train.index)
val_idx = set(X_val.index)
test_idx = set(X_test.index)
assert len(train_idx & val_idx) == 0, "LEAKAGE: train/val overlap!"
assert len(train_idx & test_idx) == 0, "LEAKAGE: train/test overlap!"
assert len(val_idx & test_idx) == 0, "LEAKAGE: val/test overlap!"
print("No index overlap between train, val, and test sets. No leakage detected.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. FEATURE ENGINEERING EXPERIMENTS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: FEATURE ENGINEERING")
print("=" * 60)

# 4.1 Feature importance (Random Forest)
print("\n--- 4.1 Random Forest Feature Importance ---")
rf_imp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_imp.fit(X_train, y_train)
importances = pd.Series(rf_imp.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
importances.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_title('Random Forest Feature Importance', fontweight='bold')
ax.set_ylabel('Importance')
plt.tight_layout()
plt.savefig('images/06_rf_feature_importance.png', dpi=150)
plt.close()
print(f"Top 10:\n{importances.head(10)}")

# 4.2 Mutual information scores
print("\n--- 4.2 Mutual Information Scores ---")
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
mi_series.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
ax.set_title('Mutual Information Scores', fontweight='bold')
ax.set_ylabel('MI Score')
plt.tight_layout()
plt.savefig('images/07_mutual_information.png', dpi=150)
plt.close()
print(f"Top 10:\n{mi_series.head(10)}")

# 4.3 Recursive Feature Elimination (RFE)
print("\n--- 4.3 RFE with Random Forest ---")
rfe_model = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(rfe_model, n_features_to_select=15, step=1)
rfe.fit(X_train, y_train)
rfe_selected = X_train.columns[rfe.support_].tolist()
rfe_ranking = pd.Series(rfe.ranking_, index=X_train.columns).sort_values()
print(f"RFE selected features (top 15): {rfe_selected}")

# 4.4 PCA analysis
print("\n--- 4.4 PCA Analysis ---")
pca = PCA(random_state=42)
pca.fit(X_train_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(cumvar)+1), cumvar, 'bo-', markersize=4)
ax.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
ax.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
n_95 = np.argmax(cumvar >= 0.95) + 1
n_90 = np.argmax(cumvar >= 0.90) + 1
ax.axvline(x=n_95, color='r', linestyle=':', alpha=0.5)
ax.axvline(x=n_90, color='orange', linestyle=':', alpha=0.5)
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('PCA: Cumulative Explained Variance', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('images/08_pca_variance.png', dpi=150)
plt.close()
print(f"Components for 90% variance: {n_90}")
print(f"Components for 95% variance: {n_95}")

# 4.5 Compare feature sets
print("\n--- 4.5 Feature Set Comparison ---")
top_15 = importances.head(15).index.tolist()
top_10 = importances.head(10).index.tolist()

feature_sets = {
    'All 30 features': X_train.columns.tolist(),
    'Top 15 (RF importance)': top_15,
    'Top 10 (RF importance)': top_10,
    'RFE 15 features': rfe_selected,
}

feat_comparison = {}
for name, feats in feature_sets.items():
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(rf_temp, X_train[feats], y_train, cv=5, scoring='accuracy')
    feat_comparison[name] = {'mean_acc': scores.mean(), 'std': scores.std()}
    print(f"  {name}: {scores.mean():.4f} +/- {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
names = list(feat_comparison.keys())
means = [feat_comparison[n]['mean_acc'] for n in names]
stds = [feat_comparison[n]['std'] for n in names]
bars = ax.bar(names, means, yerr=stds, capsize=5,
              color=['#3498db', '#2ecc71', '#e67e22', '#9b59b6'],
              edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('Cross-Validation Accuracy')
ax.set_title('Feature Set Comparison (Random Forest, 5-Fold CV)', fontweight='bold')
ax.set_ylim(min(means) - 0.02, max(means) + 0.015)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('images/09_feature_set_comparison.png', dpi=150)
plt.close()

# ============================================================
# 5. MODEL TRAINING AND COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5: MODEL TRAINING AND COMPARISON")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    print(f"\n--- Training: {name} ---")
    if name in ['Logistic Regression', 'SVM (RBF)', 'KNN']:
        X_tr, X_v = X_train_scaled, X_val_scaled
    else:
        X_tr, X_v = X_train.values, X_val.values

    # Cross-validation on training set
    scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='accuracy')
    print(f"  CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    # Fit on full training set, evaluate on validation
    model.fit(X_tr, y_train)
    val_preds = model.predict(X_v)
    val_probs = model.predict_proba(X_v)[:, 1]
    val_acc = accuracy_score(y_val, val_preds)
    print(f"  Validation Accuracy: {val_acc:.4f}")

    cv_results[name] = {
        'model': model,
        'cv_mean': scores.mean(),
        'cv_std': scores.std(),
        'val_acc': val_acc,
        'val_preds': val_preds,
        'val_probs': val_probs,
    }

# Cross-validation comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
model_names = list(cv_results.keys())
cv_means = [cv_results[n]['cv_mean'] for n in model_names]
cv_stds = [cv_results[n]['cv_std'] for n in model_names]
colors_models = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']
bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5,
              color=colors_models, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, cv_means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('5-Fold CV Accuracy')
ax.set_title('Model Comparison: Cross-Validation Accuracy', fontweight='bold')
ax.set_ylim(min(cv_means) - 0.03, max(cv_means) + 0.02)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('images/10_model_cv_comparison.png', dpi=150)
plt.close()

# ============================================================
# 6. FINAL EVALUATION ON TEST SET
# ============================================================
print("\n" + "=" * 60)
print("SECTION 6: FINAL EVALUATION ON TEST SET")
print("=" * 60)

all_metrics = {}

for name in model_names:
    model = cv_results[name]['model']
    if name in ['Logistic Regression', 'SVM (RBF)', 'KNN']:
        X_te = X_test_scaled
    else:
        X_te = X_test.values

    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    f2 = fbeta_score(y_test, preds, beta=2)
    mcc = matthews_corrcoef(y_test, preds)
    roc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    all_metrics[name] = {
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'F2': round(f2, 4),
        'MCC': round(mcc, 4),
        'ROC-AUC': round(roc, 4),
        'PR-AUC': round(pr_auc, 4),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'FPR': round(fpr, 4),
        'FNR': round(fnr, 4),
    }

    print(f"\n--- {name} ---")
    for k, v in all_metrics[name].items():
        print(f"  {k}: {v}")

# Save metrics to JSON
with open('metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print("\nMetrics saved to metrics.json")

# Metrics comparison table
metrics_df = pd.DataFrame(all_metrics).T
print(f"\n{'='*80}")
print("FULL METRICS COMPARISON TABLE")
print(f"{'='*80}")
print(metrics_df.to_string())

# ============================================================
# 7. CONFUSION MATRICES
# ============================================================
print("\n" + "=" * 60)
print("SECTION 7: CONFUSION MATRICES")
print("=" * 60)

fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))
for idx, name in enumerate(model_names):
    m = all_metrics[name]
    cm = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit (0)', 'Phish (1)'],
                yticklabels=['Legit (0)', 'Phish (1)'])
    ax.set_title(f'{name}\nAcc={m["Accuracy"]:.3f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.suptitle('Confusion Matrices — All Models (Test Set)', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('images/11_confusion_matrices.png', dpi=150)
plt.close()

# ============================================================
# 8. ROC CURVES
# ============================================================
print("\n" + "=" * 60)
print("SECTION 8: ROC AND PR CURVES")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for name in model_names:
    model = cv_results[name]['model']
    if name in ['Logistic Regression', 'SVM (RBF)', 'KNN']:
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        probs = model.predict_proba(X_test.values)[:, 1]

    fpr_curve, tpr_curve, _ = roc_curve(y_test, probs)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)

    ax1.plot(fpr_curve, tpr_curve, label=f'{name} (AUC={all_metrics[name]["ROC-AUC"]:.3f})')
    ax2.plot(rec_curve, prec_curve, label=f'{name} (AP={all_metrics[name]["PR-AUC"]:.3f})')

ax1.plot([0,1], [0,1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves — All Models', fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves — All Models', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/12_roc_pr_curves.png', dpi=150)
plt.close()

# ============================================================
# 9. THRESHOLD SELECTION (Best model — Random Forest)
# ============================================================
print("\n" + "=" * 60)
print("SECTION 9: THRESHOLD SELECTION")
print("=" * 60)

best_model_name = 'Random Forest'
best_model = cv_results[best_model_name]['model']
val_probs_best = cv_results[best_model_name]['val_probs']

thresholds = np.arange(0.1, 0.91, 0.05)
thresh_results = []
for t in thresholds:
    preds_t = (val_probs_best >= t).astype(int)
    p = precision_score(y_val, preds_t, zero_division=0)
    r = recall_score(y_val, preds_t, zero_division=0)
    f = f1_score(y_val, preds_t, zero_division=0)
    f2_val = fbeta_score(y_val, preds_t, beta=2, zero_division=0)
    thresh_results.append({'threshold': t, 'precision': p, 'recall': r, 'f1': f, 'f2': f2_val})

thresh_df = pd.DataFrame(thresh_results)
best_f2_idx = thresh_df['f2'].idxmax()
optimal_threshold = thresh_df.loc[best_f2_idx, 'threshold']
print(f"Optimal threshold (max F2 on validation): {optimal_threshold:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresh_df['threshold'], thresh_df['precision'], 'b-o', markersize=4, label='Precision')
ax.plot(thresh_df['threshold'], thresh_df['recall'], 'r-s', markersize=4, label='Recall')
ax.plot(thresh_df['threshold'], thresh_df['f1'], 'g-^', markersize=4, label='F1')
ax.plot(thresh_df['threshold'], thresh_df['f2'], 'm-D', markersize=4, label='F2')
ax.axvline(x=optimal_threshold, color='black', linestyle='--', label=f'Optimal={optimal_threshold:.2f}')
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Score')
ax.set_title('Threshold Selection on Validation Set (Random Forest)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/13_threshold_selection.png', dpi=150)
plt.close()

# Re-evaluate on test set with optimal threshold
test_probs_best = best_model.predict_proba(X_test.values)[:, 1]
test_preds_tuned = (test_probs_best >= optimal_threshold).astype(int)
test_preds_default = best_model.predict(X_test.values)

print(f"\nRandom Forest @ default threshold (0.5):")
print(f"  Recall: {recall_score(y_test, test_preds_default):.4f}, FNR: {1 - recall_score(y_test, test_preds_default):.4f}")
print(f"\nRandom Forest @ optimal threshold ({optimal_threshold:.2f}):")
print(f"  Recall: {recall_score(y_test, test_preds_tuned):.4f}, FNR: {1 - recall_score(y_test, test_preds_tuned):.4f}")

# ============================================================
# 10. ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 10: FALSE POSITIVE / FALSE NEGATIVE ANALYSIS")
print("=" * 60)

errors_df = X_test.copy()
errors_df['actual'] = y_test.values
errors_df['predicted'] = test_preds_default
errors_df['prob_phishing'] = test_probs_best

fn_df = errors_df[(errors_df['actual'] == 1) & (errors_df['predicted'] == 0)]
fp_df = errors_df[(errors_df['actual'] == 0) & (errors_df['predicted'] == 1)]
correct_df = errors_df[errors_df['actual'] == errors_df['predicted']]

print(f"\nTotal test samples: {len(errors_df)}")
print(f"False Negatives (Phishing MISSED): {len(fn_df)}")
print(f"False Positives (Legit BLOCKED):   {len(fp_df)}")
print(f"Correct predictions:               {len(correct_df)}")

# Feature means comparison: correct vs errors
print(f"\n--- Feature means: FN vs correctly classified phishing ---")
correct_phish = errors_df[(errors_df['actual'] == 1) & (errors_df['predicted'] == 1)]
if len(fn_df) > 0 and len(correct_phish) > 0:
    fn_means = fn_df[feature_cols].mean()
    correct_phish_means = correct_phish[feature_cols].mean()
    diff_fn = (fn_means - correct_phish_means).abs().sort_values(ascending=False)
    print(f"Top features where FN differ from correctly classified phishing:")
    for feat, val in diff_fn.head(10).items():
        print(f"  {feat}: diff={val:.4f} (FN mean={fn_means[feat]:.3f}, correct mean={correct_phish_means[feat]:.3f})")

print(f"\n--- Feature means: FP vs correctly classified legitimate ---")
correct_legit = errors_df[(errors_df['actual'] == 0) & (errors_df['predicted'] == 0)]
if len(fp_df) > 0 and len(correct_legit) > 0:
    fp_means = fp_df[feature_cols].mean()
    correct_legit_means = correct_legit[feature_cols].mean()
    diff_fp = (fp_means - correct_legit_means).abs().sort_values(ascending=False)
    print(f"Top features where FP differ from correctly classified legitimate:")
    for feat, val in diff_fp.head(10).items():
        print(f"  {feat}: diff={val:.4f} (FP mean={fp_means[feat]:.3f}, correct mean={correct_legit_means[feat]:.3f})")

# Error analysis visualization
if len(fn_df) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    top_diff_features = diff_fn.head(6).index.tolist() if len(fn_df) > 0 else feature_cols[:6]
    for idx, feat in enumerate(top_diff_features):
        ax = axes[idx // 3][idx % 3]
        data_plot = pd.DataFrame({
            'Correct Phishing': correct_phish[feat].value_counts(normalize=True),
            'False Negatives': fn_df[feat].value_counts(normalize=True)
        }).fillna(0)
        data_plot.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
        ax.set_title(f'{feat}', fontsize=10)
        ax.set_ylabel('Proportion')
        ax.tick_params(labelsize=8)
    plt.suptitle('False Negatives vs Correctly Classified Phishing\n(Feature Distribution Comparison)',
                 fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('images/14_fn_analysis.png', dpi=150)
    plt.close()

if len(fp_df) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    top_diff_features_fp = diff_fp.head(6).index.tolist() if len(fp_df) > 0 else feature_cols[:6]
    for idx, feat in enumerate(top_diff_features_fp):
        ax = axes[idx // 3][idx % 3]
        data_plot = pd.DataFrame({
            'Correct Legitimate': correct_legit[feat].value_counts(normalize=True),
            'False Positives': fp_df[feat].value_counts(normalize=True)
        }).fillna(0)
        data_plot.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
        ax.set_title(f'{feat}', fontsize=10)
        ax.set_ylabel('Proportion')
        ax.tick_params(labelsize=8)
    plt.suptitle('False Positives vs Correctly Classified Legitimate\n(Feature Distribution Comparison)',
                 fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('images/15_fp_analysis.png', dpi=150)
    plt.close()

# Sample FN and FP examples
print(f"\n--- Sample False Negative examples (first 5) ---")
if len(fn_df) > 0:
    print(fn_df.head().to_string())

print(f"\n--- Sample False Positive examples (first 5) ---")
if len(fp_df) > 0:
    print(fp_df.head().to_string())

# ============================================================
# 11. REPRODUCE AUTHOR'S ORIGINAL EXPERIMENT
# ============================================================
print("\n" + "=" * 60)
print("SECTION 11: REPRODUCE AUTHOR'S ORIGINAL EXPERIMENT")
print("=" * 60)

from sklearn.linear_model import RidgeClassifier

author_models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVC': SVC(probability=True, random_state=42),
    'KNeighbors': KNeighborsClassifier(),
    'Logistic': LogisticRegression(max_iter=1000, random_state=42),
    'Ridge': RidgeClassifier(random_state=42),
}

# Author's reported results
author_reported = {
    'RandomForest': {'mean_test_score': 0.9711, 'std': 0.0041},
    'SVC':          {'mean_test_score': 0.9629, 'std': 0.0064},
    'KNeighbors':   {'mean_test_score': 0.9623, 'std': 0.0046},
    'Logistic':     {'mean_test_score': 0.9270, 'std': 0.0047},
    'Ridge':        {'mean_test_score': 0.9206, 'std': 0.0053},
}

# Reload original data for faithful reproduction
df_orig = pd.read_csv('data/phishingData.csv')
df_orig.columns = df_orig.columns.str.lower()
X_orig = df_orig.drop(columns=['result'])
y_orig = df_orig['result']

reproduction_results = {}
for name, model in author_models.items():
    if name in ['SVC', 'Logistic', 'KNeighbors']:
        X_for_cv = StandardScaler().fit_transform(X_orig)
    else:
        X_for_cv = X_orig.values

    if name == 'Ridge':
        scoring = 'accuracy'
    else:
        scoring = 'accuracy'

    scores = cross_val_score(model, X_for_cv, y_orig, cv=5, scoring=scoring)
    reproduction_results[name] = {
        'our_mean': round(scores.mean(), 4),
        'our_std': round(scores.std(), 4),
        'author_mean': author_reported[name]['mean_test_score'],
        'author_std': author_reported[name]['std'],
        'delta': round(abs(scores.mean() - author_reported[name]['mean_test_score']), 4),
    }
    print(f"{name}:")
    print(f"  Author: {author_reported[name]['mean_test_score']:.4f} +/- {author_reported[name]['std']:.4f}")
    print(f"  Ours:   {scores.mean():.4f} +/- {scores.std():.4f}")
    print(f"  Delta:  {reproduction_results[name]['delta']:.4f}")

with open('reproduction_results.json', 'w') as f:
    json.dump(reproduction_results, f, indent=2)

# Reproduction comparison chart
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(reproduction_results))
width = 0.35
author_means = [reproduction_results[n]['author_mean'] for n in reproduction_results]
our_means = [reproduction_results[n]['our_mean'] for n in reproduction_results]
author_stds = [reproduction_results[n]['author_std'] for n in reproduction_results]
our_stds = [reproduction_results[n]['our_std'] for n in reproduction_results]

bars1 = ax.bar(x_pos - width/2, author_means, width, yerr=author_stds, capsize=4,
               label="Author's Reported", color='#3498db', edgecolor='black', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, our_means, width, yerr=our_stds, capsize=4,
               label="Our Reproduction", color='#e67e22', edgecolor='black', alpha=0.8)
ax.set_ylabel('Mean CV Accuracy')
ax.set_title("Author's Reported vs Our Reproduced Results", fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(list(reproduction_results.keys()))
ax.legend()
ax.set_ylim(0.90, 1.0)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images/16_reproduction_comparison.png', dpi=150)
plt.close()

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 60)
print("ALL ANALYSES COMPLETE")
print("=" * 60)
print(f"Figures saved to: images/")
print(f"Metrics saved to: metrics.json")
print(f"Reproduction results saved to: reproduction_results.json")
