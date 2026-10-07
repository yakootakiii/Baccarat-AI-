import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

sys.path.append("src")
from features import features_from_history
from betting import choose_bet

bundle = joblib.load("models/model.joblib")
model, columns = bundle["model"], bundle["columns"]

app = Flask(__name__, static_folder="static")


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/predict")
def predict():
    history = request.get_json().get("history", [])
    history = [h for h in history if h in ("B", "P", "T")]
    X = pd.DataFrame([features_from_history(history)])[columns]
    proba = model.predict_proba(X)[0]
    probs = {c: float(p) for c, p in zip(model.classes_, proba)}
    bet, ev = choose_bet(probs)
    return jsonify({"probs": probs, "bet": bet,
                    "ev": {k: float(v) for k, v in ev.items()},
                    "model": bundle["name"]})


if __name__ == "__main__":
    app.run(debug=True)
