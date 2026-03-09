"""
Titanic Survival Prediction - Flask Backend API
===============================================
Run: python app.py
API available at: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory, render_template_string
import json
import numpy as np
import os

app = Flask(__name__, static_folder="static")

# ─────────────────────────────────────────────
# Load model weights on startup
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "static/js/model_weights.json")
STATS_PATH = os.path.join(os.path.dirname(__file__), "static/js/data_stats.json")

with open(MODEL_PATH) as f:
    MODEL = json.load(f)

with open(STATS_PATH) as f:
    STATS = json.load(f)

print(f"✅ Model loaded: {MODEL['n_trees']} trees | Accuracy: {MODEL['accuracy']:.2%}")


# ─────────────────────────────────────────────
# Random Forest Prediction (Python)
# ─────────────────────────────────────────────
def predict_tree(tree, features):
    """Traverse a single decision tree."""
    node = tree
    while not node.get("leaf"):
        feat_val = features.get(node["feature"], 0)
        node = node["left"] if feat_val <= node["threshold"] else node["right"]
    return node["prob"]


def rf_predict(passenger_data):
    """Run full Random Forest prediction."""
    import math

    # Feature engineering
    sex_encoded = 1 if passenger_data.get("sex", "male") == "female" else 0
    age = float(passenger_data.get("age", MODEL["feature_medians"]["age"]))
    pclass = int(passenger_data.get("pclass", 3))
    fare = float(passenger_data.get("fare", MODEL["feature_medians"]["fare_log"]))
    sibsp = int(passenger_data.get("sibsp", 0))
    parch = int(passenger_data.get("parch", 0))
    embarked_map = {"S": 0, "Q": 1, "C": 2}
    embarked = embarked_map.get(passenger_data.get("embarked", "S"), 0)

    # Derived features
    age_group = 0 if age < 12 else 1 if age < 18 else 2 if age < 35 else 3 if age < 60 else 4
    fare_log = math.log1p(fare)
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    features = {
        "pclass": pclass,
        "sex_encoded": sex_encoded,
        "age": age,
        "age_group": age_group,
        "fare_log": fare_log,
        "family_size": family_size,
        "is_alone": is_alone,
        "sibsp": sibsp,
        "parch": parch,
        "embarked_encoded": embarked
    }

    # Average across all trees
    probs = [predict_tree(tree, features) for tree in MODEL["trees"]]
    survival_prob = float(np.mean(probs))
    survived = survival_prob >= 0.5

    return {
        "survival_probability": round(survival_prob, 4),
        "survived": survived,
        "confidence": round(abs(survival_prob - 0.5) * 2, 4),
        "features_used": features
    }


# ─────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body: {
        "pclass": 1|2|3,
        "sex": "male"|"female",
        "age": float,
        "fare": float,
        "sibsp": int,
        "parch": int,
        "embarked": "S"|"C"|"Q"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    result = rf_predict(data)
    return jsonify({
        "success": True,
        "prediction": result,
        "model_info": {
            "type": MODEL["model_type"],
            "n_trees": MODEL["n_trees"],
            "accuracy": MODEL["accuracy"],
            "auc": MODEL["auc"]
        }
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return dataset statistics for charts."""
    return jsonify({"success": True, "stats": STATS})


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model metadata."""
    return jsonify({
        "success": True,
        "model": {
            "type": MODEL["model_type"],
            "n_trees": MODEL["n_trees"],
            "features": MODEL["features"],
            "accuracy": MODEL["accuracy"],
            "auc": MODEL["auc"],
            "feature_importance": STATS.get("feature_importance", {})
        }
    })


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    """
    POST /api/batch-predict
    Body: { "passengers": [...] }
    """
    data = request.get_json()
    if not data or "passengers" not in data:
        return jsonify({"error": "Provide 'passengers' array"}), 400

    results = []
    for passenger in data["passengers"]:
        result = rf_predict(passenger)
        results.append(result)

    return jsonify({
        "success": True,
        "count": len(results),
        "predictions": results
    })


if __name__ == "__main__":
    print("🚢 Titanic Survival Prediction API")
    print("=" * 40)
    print("  Server: http://localhost:5000")
    print("  API:    http://localhost:5000/api/predict")
    print("  Stats:  http://localhost:5000/api/stats")
    print("=" * 40)
    app.run(debug=True, host="0.0.0.0", port=5000)
