import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Load data
df = pd.read_csv('data/phishingData.csv')
df.columns = df.columns.str.lower()

# 1. Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='result')
plt.title('Distribution of Target Variable (Result)')
plt.savefig('images/class_distribution.png', bbox_inches='tight')
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(15, 12))
corr = df.corr(method='spearman')
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Spearman Correlation Heatmap')
plt.savefig('images/correlation_heatmap.png', bbox_inches='tight')
plt.close()

# 3. Crosstab Analysis (URL Length vs Result)
ct = pd.crosstab(df['url_length'], df['result'], normalize='index')
ct.plot(kind='bar', stacked=True, figsize=(8,5))
plt.title('URL Length vs Phishing Result')
plt.ylabel('Proportion')
plt.savefig('images/crosstab_url_length.png', bbox_inches='tight')
plt.close()

# Modeling setup
X = df.drop(columns=['result'])
y = df['result'].map({-1: 0, 1: 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Confusion Matrix
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

cm_rf = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('images/cm_rf.png', bbox_inches='tight')
plt.close()

# Logistic Regression Confusion Matrix
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

cm_lr = confusion_matrix(y_test, lr_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('images/cm_lr.png', bbox_inches='tight')
plt.close()

print("Images generated successfully!")
