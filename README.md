# 🚢 Titanic Survival Predictor

A complete end-to-end machine learning project — Random Forest classifier with an interactive webpage, Flask API, and full EDA.

---

## 📁 Project Structure

```
titanic-project/
├── index.html                   ← Interactive prediction webpage (open directly)
├── app.py                       ← Flask backend API
├── requirements.txt
├── model/
│   ├── train_model.py           ← Full ML training script with EDA & plots
│   ├── train_and_export.py      ← Lightweight export for JS weights
│   └── titanic_data.csv         ← Dataset (auto-generated / replace with Kaggle data)
└── static/
    └── js/
        ├── model_weights.json   ← Random Forest trees (80 trees, exported for JS)
        └── data_stats.json      ← Dataset stats for visualization charts
```

---

## 🚀 Quick Start

### Option A — Open webpage directly (no server needed)
```bash
# Just open index.html in your browser
open index.html
# The model runs entirely in JavaScript!
```

### Option B — Run with Flask backend
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates model_weights.json)
cd model/
python train_and_export.py

# 3. Start the Flask server
cd ..
python app.py

# 4. Open http://localhost:5000
```

---

## 📊 Features

| Feature | Description |
|---|---|
| **Interactive Predictor** | Enter passenger details, get survival probability instantly |
| **Random Forest (JS)** | 80 decision trees running in-browser — no server needed |
| **EDA Charts** | Survival by sex, class, age, embarkation port |
| **Flask API** | REST endpoints for prediction, batch prediction, stats |
| **Historical Profiles** | Test with "Rose-type", "Jack-type" and more |
| **Model Info Tab** | Feature importances, metrics, architecture details |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Predict survival for one passenger |
| POST | `/api/batch-predict` | Predict for multiple passengers |
| GET | `/api/stats` | Dataset statistics |
| GET | `/api/model-info` | Model metadata & feature importances |

### Example API Call
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"pclass":1,"sex":"female","age":25,"fare":80,"sibsp":0,"parch":0,"embarked":"C"}'
```

---

## 🤖 Model Details

- **Algorithm**: Random Forest Classifier
- **Trees**: 80 decision trees
- **Max Depth**: 7
- **Features**: 10 engineered features
- **Test Accuracy**: ~79%
- **ROC-AUC**: ~0.85

### Features Used
- `pclass` — Passenger class (1/2/3)
- `sex_encoded` — Sex (0=male, 1=female)
- `age` — Age in years
- `age_group` — Age bucket (child/teen/adult/middle-aged/elderly)
- `fare_log` — Log-transformed fare
- `family_size` — Total family members aboard
- `is_alone` — Travelling alone (boolean)
- `sibsp` — Siblings/spouses aboard
- `parch` — Parents/children aboard
- `embarked_encoded` — Embarkation port

---

## 📈 EDA Insights

- **Women survived at 74%** vs men at 19%
- **1st class** had the highest survival rate (63%)
- **3rd class** had the lowest survival rate (24%)
- **Children under 12** had better survival odds
- **Cherbourg** passengers had higher survival than Southampton/Queenstown
- Passengers **travelling with small families** had better odds than those alone or in large groups

---

## 📦 Dataset

Uses the classic Titanic dataset. You can replace `model/titanic_data.csv` with the original Kaggle dataset:
https://www.kaggle.com/c/titanic/data
