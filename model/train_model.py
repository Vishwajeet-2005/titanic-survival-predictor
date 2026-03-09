"""
Titanic Survival Prediction - ML Training Script
================================================
Trains a Random Forest model on the Titanic dataset.
Outputs: model weights as JSON for embedding in JavaScript.

Usage:
    pip install pandas scikit-learn numpy matplotlib seaborn
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load & Explore Data
# ─────────────────────────────────────────────
print("=" * 60)
print("  TITANIC SURVIVAL PREDICTION - MODEL TRAINING")
print("=" * 60)

# Using the Titanic dataset (classic ML benchmark)
# Download from: https://www.kaggle.com/c/titanic/data
# Or use seaborn's built-in dataset:
import seaborn as sns
df = sns.load_dataset("titanic")

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n🔍 First 5 rows:\n{df.head()}")
print(f"\n📋 Info:\n")
df.info()
print(f"\n📈 Statistics:\n{df.describe()}")
print(f"\n❓ Missing Values:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────
# 2. Exploratory Data Analysis
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Survival rate
survival_rate = df["survived"].mean()
print(f"\n✅ Overall Survival Rate: {survival_rate:.2%}")

# By Sex
print("\n👤 Survival by Sex:")
print(df.groupby("sex")["survived"].mean().to_string())

# By Pclass
print("\n🎫 Survival by Passenger Class:")
print(df.groupby("pclass")["survived"].mean().to_string())

# By Embarked
print("\n⚓ Survival by Embarkation Port:")
print(df.groupby("embark_town")["survived"].mean().to_string())

# Age stats
print(f"\n🎂 Age Stats:")
print(f"   Min: {df['age'].min():.0f}  Max: {df['age'].max():.0f}  Mean: {df['age'].mean():.1f}")

# ─────────────────────────────────────────────
# 3. EDA Visualizations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Titanic Survival - Exploratory Data Analysis", fontsize=18, fontweight="bold", y=1.01)
palette = {"survived": "#2ecc71", "dead": "#e74c3c"}
SURVIVED_PALETTE = {0: "#e74c3c", 1: "#2ecc71"}

# 3.1 Survival Count
ax = axes[0, 0]
counts = df["survived"].value_counts()
bars = ax.bar(["Did Not Survive", "Survived"], counts.values,
              color=["#e74c3c", "#2ecc71"], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha='center', fontsize=12, fontweight='bold')
ax.set_title("Overall Survival", fontsize=13, fontweight='bold')
ax.set_ylabel("Count")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3.2 Survival by Sex
ax = axes[0, 1]
sex_survival = df.groupby("sex")["survived"].mean() * 100
bars = ax.bar(sex_survival.index, sex_survival.values,
              color=["#3498db", "#e91e8c"], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, sex_survival.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha='center', fontsize=11, fontweight='bold')
ax.set_title("Survival Rate by Sex", fontsize=13, fontweight='bold')
ax.set_ylabel("Survival Rate (%)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3.3 Survival by Passenger Class
ax = axes[0, 2]
class_survival = df.groupby("pclass")["survived"].mean() * 100
bars = ax.bar([f"Class {c}" for c in class_survival.index], class_survival.values,
              color=["#f39c12", "#95a5a6", "#7f8c8d"], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, class_survival.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha='center', fontsize=11, fontweight='bold')
ax.set_title("Survival Rate by Class", fontsize=13, fontweight='bold')
ax.set_ylabel("Survival Rate (%)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3.4 Age Distribution
ax = axes[1, 0]
survived = df[df["survived"] == 1]["age"].dropna()
not_survived = df[df["survived"] == 0]["age"].dropna()
ax.hist(not_survived, bins=30, alpha=0.7, color="#e74c3c", label="Did Not Survive", edgecolor='white')
ax.hist(survived, bins=30, alpha=0.7, color="#2ecc71", label="Survived", edgecolor='white')
ax.set_title("Age Distribution by Survival", fontsize=13, fontweight='bold')
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3.5 Fare Distribution (log scale)
ax = axes[1, 1]
df_clean = df.dropna(subset=["fare"])
survived_fare = df_clean[df_clean["survived"] == 1]["fare"]
not_survived_fare = df_clean[df_clean["survived"] == 0]["fare"]
ax.hist(np.log1p(not_survived_fare), bins=30, alpha=0.7, color="#e74c3c", label="Did Not Survive", edgecolor='white')
ax.hist(np.log1p(survived_fare), bins=30, alpha=0.7, color="#2ecc71", label="Survived", edgecolor='white')
ax.set_title("Fare Distribution (log scale)", fontsize=13, fontweight='bold')
ax.set_xlabel("Log(Fare + 1)")
ax.set_ylabel("Count")
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3.6 Heatmap: Class x Sex Survival
ax = axes[1, 2]
pivot = df.pivot_table(values="survived", index="pclass", columns="sex", aggfunc="mean")
sns.heatmap(pivot, annot=True, fmt=".2%", cmap="RdYlGn",
            ax=ax, linewidths=0.5, cbar_kws={"label": "Survival Rate"})
ax.set_title("Survival Rate: Class × Sex", fontsize=13, fontweight='bold')
ax.set_xlabel("Sex")
ax.set_ylabel("Passenger Class")

# 3.7 Embarkation Port Survival
ax = axes[2, 0]
embark_survival = df.groupby("embark_town")["survived"].mean() * 100
embark_survival = embark_survival.dropna()
colors_embark = ["#9b59b6", "#1abc9c", "#e67e22"]
bars = ax.bar(embark_survival.index, embark_survival.values,
              color=colors_embark[:len(embark_survival)], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, embark_survival.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha='center', fontsize=11, fontweight='bold')
ax.set_title("Survival by Embarkation Port", fontsize=13, fontweight='bold')
ax.set_ylabel("Survival Rate (%)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3.8 Family Size vs Survival
ax = axes[2, 1]
df["family_size"] = df["sibsp"] + df["parch"] + 1
family_survival = df.groupby("family_size")["survived"].mean() * 100
ax.plot(family_survival.index, family_survival.values, 'o-',
        color="#3498db", linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2.5)
ax.fill_between(family_survival.index, family_survival.values, alpha=0.15, color="#3498db")
ax.set_title("Survival Rate by Family Size", fontsize=13, fontweight='bold')
ax.set_xlabel("Family Size (including self)")
ax.set_ylabel("Survival Rate (%)")
ax.set_xticks(family_survival.index)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3.9 Correlation Heatmap
ax = axes[2, 2]
numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, center=0)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("eda_analysis.png", dpi=150, bbox_inches='tight', facecolor='white')
print("\n✅ EDA plots saved as 'eda_analysis.png'")
plt.close()

# ─────────────────────────────────────────────
# 4. Feature Engineering
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FEATURE ENGINEERING")
print("=" * 60)

def prepare_features(df):
    df = df.copy()

    # Sex encoding
    df["sex_encoded"] = (df["sex"] == "female").astype(int)

    # Age: fill with median by class & sex
    df["age"] = df.groupby(["pclass", "sex"])["age"].transform(lambda x: x.fillna(x.median()))
    df["age"] = df["age"].fillna(df["age"].median())

    # Age buckets
    df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 100],
                             labels=[0, 1, 2, 3, 4]).astype(float)

    # Fare: fill with median
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["fare_log"] = np.log1p(df["fare"])

    # Family features
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    # Embarkation
    embark_map = {"S": 0, "Q": 1, "C": 2}
    df["embarked_encoded"] = df["embarked"].map(embark_map).fillna(0)

    # Title from name (if available)
    if "name" in df.columns:
        df["title"] = df["name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        title_map = {
            "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3,
            "Dr": 4, "Rev": 4, "Col": 4, "Major": 4,
            "Mlle": 1, "Countess": 2, "Ms": 1, "Lady": 2,
            "Jonkheer": 4, "Don": 4, "Dona": 2, "Mme": 2, "Capt": 4, "Sir": 4
        }
        df["title_encoded"] = df["title"].map(title_map).fillna(4)
    else:
        df["title_encoded"] = 4

    return df

df_processed = prepare_features(df)
FEATURES = ["pclass", "sex_encoded", "age", "age_group", "fare_log",
            "family_size", "is_alone", "sibsp", "parch", "embarked_encoded", "title_encoded"]

print(f"\n✅ Features used: {FEATURES}")

X = df_processed[FEATURES]
y = df_processed["survived"]

# Remove rows where y is NaN
mask = ~y.isna()
X, y = X[mask], y[mask]

print(f"   Training samples: {len(X)}")
print(f"   Feature matrix shape: {X.shape}")

# ─────────────────────────────────────────────
# 5. Train / Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 6. Train Models
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL TRAINING")
print("=" * 60)

# Random Forest (primary)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

# Logistic Regression (comparison)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)

# ─────────────────────────────────────────────
# 7. Evaluate Models
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL EVALUATION")
print("=" * 60)

for name, model, X_t in [("Random Forest", rf, X_test), ("Logistic Regression", lr, X_test_scaled)]:
    preds = model.predict(X_t)
    proba = model.predict_proba(X_t)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    cv = cross_val_score(model,
                         X if name == "Random Forest" else scaler.transform(X),
                         y, cv=StratifiedKFold(5), scoring='accuracy').mean()
    print(f"\n🔹 {name}")
    print(f"   Accuracy:       {acc:.4f} ({acc:.2%})")
    print(f"   ROC-AUC:        {auc:.4f}")
    print(f"   CV Accuracy:    {cv:.4f}")
    print(f"\n   Classification Report:\n{classification_report(y_test, preds, target_names=['Did Not Survive', 'Survived'])}")

# ─────────────────────────────────────────────
# 8. Feature Importance
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FEATURE IMPORTANCES (Random Forest)")
print("=" * 60)

importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(importances.to_string())

# ─────────────────────────────────────────────
# 9. Export Model Weights for JavaScript
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EXPORTING MODEL FOR JAVASCRIPT")
print("=" * 60)

# Export Random Forest trees in a simplified format
# We'll export the tree structure for use in JS

def export_tree(tree, feature_names):
    """Export a single decision tree as a nested dict."""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "leaf"
        for i in tree_.feature
    ]

    def recurse(node):
        if tree_.feature[node] == -2:  # Leaf
            values = tree_.value[node][0]
            total = values.sum()
            prob = float(values[1] / total) if total > 0 else 0.5
            return {"leaf": True, "prob": round(prob, 4)}
        return {
            "feature": feature_name[node],
            "threshold": round(float(tree_.threshold[node]), 4),
            "left": recurse(tree_.children_left[node]),
            "right": recurse(tree_.children_right[node])
        }
    return recurse(0)

# Export first 50 trees (balance between accuracy & file size)
N_TREES = 50
trees_export = []
for i in range(min(N_TREES, len(rf.estimators_))):
    trees_export.append(export_tree(rf.estimators_[i], FEATURES))

model_data = {
    "model_type": "RandomForest",
    "features": FEATURES,
    "n_trees": len(trees_export),
    "trees": trees_export,
    "feature_medians": {
        feat: float(X[feat].median()) for feat in FEATURES
    },
    "accuracy": float(accuracy_score(y_test, rf.predict(X_test))),
    "auc": float(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
}

with open("../static/js/model_weights.json", "w") as f:
    json.dump(model_data, f, indent=2)

print(f"✅ Model exported: {N_TREES} trees → static/js/model_weights.json")
print(f"   File size: ~{len(json.dumps(model_data)) / 1024:.0f} KB")
print(f"   Model Accuracy: {model_data['accuracy']:.2%}")
print(f"   ROC-AUC: {model_data['auc']:.4f}")

# ─────────────────────────────────────────────
# 10. Model Performance Plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Random Forest Model Performance", fontsize=16, fontweight='bold')

# Confusion Matrix
ax = axes[0]
cm = confusion_matrix(y_test, rf.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted Dead', 'Predicted Survived'],
            yticklabels=['Actually Dead', 'Actually Survived'])
ax.set_title("Confusion Matrix", fontsize=13, fontweight='bold')

# Feature Importance
ax = axes[1]
top_features = importances.head(8)
colors_feat = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
bars = ax.barh(top_features.index[::-1], top_features.values[::-1], color=colors_feat)
ax.set_title("Top Feature Importances", fontsize=13, fontweight='bold')
ax.set_xlabel("Importance Score")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ROC Curve
from sklearn.metrics import roc_curve
ax = axes[2]
fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
auc_val = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
ax.plot(fpr, tpr, color="#2ecc71", lw=2.5, label=f"RF (AUC = {auc_val:.3f})")
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr.predict_proba(X_test_scaled)[:, 1])
auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
ax.plot(fpr_lr, tpr_lr, color="#3498db", lw=2, linestyle='--', label=f"LR (AUC = {auc_lr:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.fill_between(fpr, tpr, alpha=0.1, color="#2ecc71")
ax.set_title("ROC Curve Comparison", fontsize=13, fontweight='bold')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("model_performance.png", dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Performance plots saved as 'model_performance.png'")
plt.close()

print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print("=" * 60)
print("  Files generated:")
print("  📊 eda_analysis.png       - EDA visualizations")
print("  📈 model_performance.png  - Model evaluation charts")
print("  🤖 static/js/model_weights.json - JS model weights")
print("=" * 60)
