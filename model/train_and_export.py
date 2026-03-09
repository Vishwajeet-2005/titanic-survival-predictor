"""
Run this script to retrain the model using the real Titanic dataset.
Place titanic_data.csv in the same folder as this script.

Usage:
    cd model/
    python train_and_export.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import json, os, re

# ── Load real Titanic CSV ──────────────────────────────
df = pd.read_csv('titanic_data.csv')   # ← real Kaggle dataset
print(f"Loaded {len(df)} passengers | Survival: {df.Survived.mean():.2%}")

# ── Feature Engineering ────────────────────────────────
def extract_title(name):
    m = re.search(r' ([A-Za-z]+)\.', name)
    title = m.group(1) if m else 'Mr'
    rare = {'Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'}
    if title in rare: return 'Rare'
    if title == 'Mlle': return 'Miss'
    if title in ('Ms','Mme'): return 'Mrs'
    return title

title_map = {'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4}
df['title_encoded'] = df['Name'].apply(lambda x: title_map.get(extract_title(x), 0))
df['sex_encoded']   = (df['Sex'] == 'female').astype(int)
df['age']           = df.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
df['age']           = df['age'].fillna(df['age'].median())
df['age_group']     = pd.cut(df['age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(float)
df['fare']          = df['Fare'].fillna(df['Fare'].median())
df['fare_log']      = np.log1p(df['fare'])
df['family_size']   = df['SibSp'] + df['Parch'] + 1
df['is_alone']      = (df['family_size'] == 1).astype(int)
df['embarked_encoded'] = df['Embarked'].map({'S':0,'Q':1,'C':2}).fillna(0)

FEATURES = ["pclass","sex_encoded","age","age_group","fare_log",
            "family_size","is_alone","sibsp","parch","embarked_encoded","title_encoded"]

col_map = {'pclass':'Pclass','sibsp':'SibSp','parch':'Parch'}
for f in FEATURES:
    if f not in df.columns and f in col_map:
        df[f] = df[col_map[f]]

X = df[FEATURES].copy()
X.columns = FEATURES
y = df['Survived']
mask = ~y.isna()
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── Train ──────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=80, max_depth=7, min_samples_split=4,
                             min_samples_leaf=2, max_features='sqrt', random_state=42)
rf.fit(X_train, y_train)

acc = accuracy_score(y_test, rf.predict(X_test))
auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")

# ── Export trees to JSON ───────────────────────────────
def export_tree(tree, feature_names):
    t = tree.tree_
    fname = [feature_names[i] if i!=-2 else "leaf" for i in t.feature]
    def recurse(node):
        if t.feature[node] == -2:
            vals = t.value[node][0]; total = vals.sum()
            return {"leaf":True,"prob":round(float(vals[1]/total),4) if total>0 else 0.5}
        return {"feature":fname[node],"threshold":round(float(t.threshold[node]),4),
                "left":recurse(t.children_left[node]),"right":recurse(t.children_right[node])}
    return recurse(0)

trees_export = [export_tree(rf.estimators_[i], FEATURES) for i in range(len(rf.estimators_))]

model_data = {
    "model_type": "RandomForest",
    "features": FEATURES,
    "n_trees": len(trees_export),
    "trees": trees_export,
    "feature_medians": {f: float(X[f].median()) for f in FEATURES},
    "accuracy": round(acc,4),
    "auc": round(auc,4)
}

out = '../static/js/model_weights.json'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out,'w') as f: json.dump(model_data, f)
print(f"✅ Exported {len(trees_export)} trees → {out}")

# ── Export stats ───────────────────────────────────────
age_bins_labels = ['0-12','13-18','19-35','36-60','60+']
age_bins_vals = []
for lo,hi in [(0,12),(13,18),(19,35),(36,60),(60,120)]:
    subset = df[(df.age>=lo)&(df.age<hi)&(~df.Survived.isna())]
    age_bins_vals.append(round(float(subset.Survived.mean()),4) if len(subset)>0 else 0)

importance = dict(zip(FEATURES,[round(float(x),4) for x in rf.feature_importances_]))
stats = {
    "survival_rate": round(float(df.Survived.mean()),4),
    "total": int(len(df)),
    "survived_count": int(df.Survived.sum()),
    "dead_count": int((df.Survived==0).sum()),
    "by_sex": {k:round(float(v),4) for k,v in df.groupby('Sex')['Survived'].mean().items()},
    "by_class": {str(k):round(float(v),4) for k,v in df.groupby('Pclass')['Survived'].mean().items()},
    "by_embarked": {k:round(float(v),4) for k,v in df.groupby('Embarked')['Survived'].mean().items() if k},
    "age_bins": {"labels": age_bins_labels, "values": age_bins_vals},
    "feature_importance": importance
}
with open('../static/js/data_stats.json','w') as f: json.dump(stats, f)
print("✅ Exported stats → ../static/js/data_stats.json")
