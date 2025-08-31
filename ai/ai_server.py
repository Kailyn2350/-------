# ai_server.py
from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

app = Flask(__name__)


class FashionAI:
    def __init__(self):
        self.models = {
            "color": RandomForestRegressor(n_estimators=50, max_depth=10),
            "fit": RandomForestRegressor(n_estimators=50, max_depth=10),
            "trend": RandomForestRegressor(n_estimators=50, max_depth=10),
            "balance": RandomForestRegressor(n_estimators=50, max_depth=10),
        }
        self.scalers = {
            "color": StandardScaler(),
            "fit": StandardScaler(),
            "trend": StandardScaler(),
            "balance": StandardScaler(),
        }

    def prepare_features(self, training_data):
        features = []
        for data in training_data:
            img_features = data["imageFeatures"]
            feature_vector = []
            # 색상 히스토그램 (48)
            feature_vector.extend(img_features["colorHistogram"]["r"])
            feature_vector.extend(img_features["colorHistogram"]["g"])
            feature_vector.extend(img_features["colorHistogram"]["b"])
            # 주요 색상 (15)
            for color in img_features["dominantColors"][:5]:
                feature_vector.extend([color["r"], color["g"], color["b"]])
            # 기타 특징 (3)
            feature_vector.extend(
                [
                    img_features["brightness"],
                    img_features["contrast"],
                    img_features["saturation"],
                ]
            )
            features.append(feature_vector)
        return np.array(features)

    def convert_feedback_to_score(self, ai_score, feedback):
        if feedback == "accurate":
            return ai_score
        if feedback == "high":
            return max(0, ai_score - 15)
        if feedback == "low":
            return min(100, ai_score + 15)
        return ai_score

    def train(self, training_data):
        if len(training_data) < 50:
            return False
        print(f"🎯 {len(training_data)}개 데이터로 AI 훈련 시작")
        X = self.prepare_features(training_data)
        for category in ["color", "fit", "trend", "balance"]:
            y = []
            for data in training_data:
                ai_score = data["aiPredictions"][category]
                feedback = data["userFeedback"][category]
                y.append(self.convert_feedback_to_score(ai_score, feedback))
            X_scaled = self.scalers[category].fit_transform(X)
            self.models[category].fit(X_scaled, np.array(y))
            print(f"✅ {category} 모델 훈련 완료")
        self.save_models()
        return True

    def save_models(self):
        os.makedirs("models", exist_ok=True)
        for category in ["color", "fit", "trend", "balance"]:
            joblib.dump(self.models[category], f"models/{category}_model.pkl")
            joblib.dump(self.scalers[category], f"models/{category}_scaler.pkl")
        print("💾 모델 저장 완료")


fashion_ai = FashionAI()

# --- 비밀 토큰 검사 추가 ---
RETRAIN_SECRET = os.environ.get("RETRAIN_SECRET")


@app.post("/retrain")
def retrain_model():
    if RETRAIN_SECRET and request.headers.get("X-RETRAIN-SECRET") != RETRAIN_SECRET:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    training_data = request.json.get("trainingData", [])
    success = fashion_ai.train(training_data)
    return (
        jsonify({"status": "success", "message": "모델 훈련 완료"})
        if success
        else jsonify({"status": "error", "message": "데이터 부족"})
    ), (200 if success else 400)


# --- Render 포트 사용 (이 블록만 유지) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
