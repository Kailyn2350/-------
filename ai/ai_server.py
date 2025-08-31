# ai_server.py
from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import os

app = Flask(__name__)


class FashionAI:
    def __init__(self):
        # 앙상블 모델: 여러 모델의 예측을 평균내어 더 정확한 결과 생성
        self.model_ensemble = {
            "color": {
                "rf": RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42),
                "gb": GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
                "ridge": Ridge(alpha=1.0, random_state=42)
            },
            "fit": {
                "rf": RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42),
                "gb": GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
                "ridge": Ridge(alpha=1.0, random_state=42)
            },
            "trend": {
                "rf": RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42),
                "gb": GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
                "ridge": Ridge(alpha=1.0, random_state=42)
            },
            "balance": {
                "rf": RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42),
                "gb": GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
                "ridge": Ridge(alpha=1.0, random_state=42)
            }
        }
        self.scalers = {
            "color": StandardScaler(),
            "fit": StandardScaler(),
            "trend": StandardScaler(),
            "balance": StandardScaler(),
        }
        self.model_weights = {
            "color": {"rf": 0.5, "gb": 0.35, "ridge": 0.15},
            "fit": {"rf": 0.5, "gb": 0.35, "ridge": 0.15},
            "trend": {"rf": 0.5, "gb": 0.35, "ridge": 0.15},
            "balance": {"rf": 0.5, "gb": 0.35, "ridge": 0.15}
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
            
            # 주요 색상 (24) - 8개 색상 * 3(RGB)
            for color in img_features["dominantColors"][:8]:
                feature_vector.extend([color["r"], color["g"], color["b"]])
            
            # 기본 특징들 (3)
            feature_vector.extend([
                img_features["brightness"],
                img_features["contrast"],
                img_features["saturation"]
            ])
            
            # 추가 특징들 (2)
            feature_vector.extend([
                img_features.get("colorVariance", 0),
                img_features.get("edgeIntensity", 0)
            ])
            
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
        if len(training_data) < 30:  # 최소 데이터 요구량 낮춤
            return False
        print(f"🎯 {len(training_data)}개 데이터로 AI 앙상블 훈련 시작")
        X = self.prepare_features(training_data)
        
        for category in ["color", "fit", "trend", "balance"]:
            y = []
            for data in training_data:
                ai_score = data["aiPredictions"][category]
                feedback = data["userFeedback"][category]
                y.append(self.convert_feedback_to_score(ai_score, feedback))
            
            X_scaled = self.scalers[category].fit_transform(X)
            y_array = np.array(y)
            
            # 각 모델별로 훈련 및 성능 평가
            model_scores = {}
            for model_name, model in self.model_ensemble[category].items():
                model.fit(X_scaled, y_array)
                # 교차 검증으로 모델 성능 평가
                cv_scores = cross_val_score(model, X_scaled, y_array, cv=3, scoring='neg_mean_squared_error')
                model_scores[model_name] = -cv_scores.mean()
                print(f"  {category} {model_name}: CV MSE = {model_scores[model_name]:.3f}")
            
            # 성능에 기반한 가중치 자동 조정
            total_score = sum(1/score for score in model_scores.values())
            for model_name, score in model_scores.items():
                self.model_weights[category][model_name] = (1/score) / total_score
            
            print(f"✅ {category} 앙상블 모델 훈련 완료")
            print(f"   가중치: {self.model_weights[category]}")
        
        self.save_models()
        return True

    def save_models(self):
        os.makedirs("models", exist_ok=True)
        for category in ["color", "fit", "trend", "balance"]:
            # 앙상블 모델들 저장
            for model_name, model in self.model_ensemble[category].items():
                joblib.dump(model, f"models/{category}_{model_name}_model.pkl")
            joblib.dump(self.scalers[category], f"models/{category}_scaler.pkl")
        # 가중치 저장
        joblib.dump(self.model_weights, "models/ensemble_weights.pkl")
        print("💾 앙상블 모델 저장 완료")
    
    def load_models(self):
        """저장된 모델 불러오기"""
        try:
            for category in ["color", "fit", "trend", "balance"]:
                for model_name in ["rf", "gb", "ridge"]:
                    model_path = f"models/{category}_{model_name}_model.pkl"
                    if os.path.exists(model_path):
                        self.model_ensemble[category][model_name] = joblib.load(model_path)
                scaler_path = f"models/{category}_scaler.pkl"
                if os.path.exists(scaler_path):
                    self.scalers[category] = joblib.load(scaler_path)
            
            weights_path = "models/ensemble_weights.pkl"
            if os.path.exists(weights_path):
                self.model_weights = joblib.load(weights_path)
            print("✅ 저장된 앙상블 모델 로드 완료")
            return True
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {e}")
            return False
    
    def predict_ensemble(self, features):
        """앙상블 모델로 예측"""
        predictions = {}
        for category in ["color", "fit", "trend", "balance"]:
            X_scaled = self.scalers[category].transform([features])
            
            # 각 모델의 예측값 계산
            model_preds = {}
            for model_name, model in self.model_ensemble[category].items():
                pred = model.predict(X_scaled)[0]
                model_preds[model_name] = max(0, min(100, pred))  # 0-100 범위 제한
            
            # 가중 평균으로 최종 예측
            weighted_pred = sum(
                pred * self.model_weights[category][model_name] 
                for model_name, pred in model_preds.items()
            )
            predictions[category] = max(40, min(99, int(weighted_pred)))
        
        # 총점 계산
        predictions['total'] = int(sum(predictions.values()) / 4)
        return predictions


fashion_ai = FashionAI()

# 서버 시작 시 기존 모델 로드
fashion_ai.load_models()

# --- 비밀 토큰 검사 추가 ---
RETRAIN_SECRET = os.environ.get("RETRAIN_SECRET")


@app.post("/predict")
def predict_fashion():
    """실시간 패션 점수 예측 API"""
    try:
        image_features = request.json.get("imageFeatures")
        if not image_features:
            return jsonify({"error": "이미지 특징이 필요합니다"}), 400
        
        # 앙상블 모델로 예측
        predictions = fashion_ai.predict_ensemble(image_features)
        return jsonify({
            "success": True,
            "predictions": predictions,
            "model_type": "ensemble",
            "confidence": "high"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/retrain")
def retrain_model():
    if RETRAIN_SECRET and request.headers.get("X-RETRAIN-SECRET") != RETRAIN_SECRET:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    training_data = request.json.get("trainingData", [])
    success = fashion_ai.train(training_data)
    return (
        jsonify({"status": "success", "message": "앙상블 모델 훈련 완료"})
        if success
        else jsonify({"status": "error", "message": "데이터 부족 (최소 30개 필요)"})
    ), (200 if success else 400)


# --- Render 포트 사용 (이 블록만 유지) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
