// server/server.js
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');


const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// ===== 환경변수 =====
const PORT = process.env.PORT || 3000;
const MONGODB_URI = process.env.MONGODB_URI;          // Atlas 연결문자열
const AI_URL = process.env.AI_URL;                    // 예: https://your-ai.onrender.com/retrain
const RETRAIN_SECRET = process.env.RETRAIN_SECRET;

// ===== MongoDB 연결 =====
mongoose.connect(MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true });

// ===== 스키마 (TTL은 Date 필드에 걸어야 동작) =====
const TrainingDataSchema = new mongoose.Schema({
    timestamp: { type: Date, default: Date.now, index: true, expires: '180d' }, // 180일 후 자동 삭제
    imageFeatures: {
        colorHistogram: { r: [Number], g: [Number], b: [Number] },
        dominantColors: [{ r: Number, g: Number, b: Number, count: Number }],
        brightness: Number,
        contrast: Number,
        saturation: Number
    },
    aiPredictions: { color: Number, fit: Number, trend: Number, balance: Number, total: Number },
    userFeedback: { color: String, fit: String, trend: String, balance: String },
    userRating: Number,
    sessionId: String
});
const TrainingData = mongoose.model('TrainingData', TrainingDataSchema);

// ===== API =====
app.post('/api/collect-training-data', async (req, res) => {
    try {
        const doc = new TrainingData(req.body);
        await doc.save();

        const totalCount = await TrainingData.countDocuments();
        if (AI_URL && totalCount % 1000 === 0) {
            triggerModelRetraining(); // 비동기 트리거
        }

        res.json({ success: true, message: '학습 데이터 수집 완료', totalSamples: totalCount });
    } catch (err) {
        console.error('학습 데이터 저장 실패:', err);
        res.status(500).json({ success: false, error: err.message });
    }
});

app.get('/api/training-stats', async (_req, res) => {
    const totalSamples = await TrainingData.countDocuments();
    const recentSamples = await TrainingData.countDocuments({
        timestamp: { $gte: new Date(Date.now() - 24 * 60 * 60 * 1000) }
    });
    res.json({ totalSamples, recentSamples });
});

// AI 예측 프록시 API
app.post('/api/ai-predict', async (req, res) => {
    try {
        if (!AI_URL) {
            return res.status(503).json({ 
                success: false, 
                error: 'AI 서버가 설정되지 않았습니다' 
            });
        }

        // AI 서버로 예측 요청 전달
        const response = await fetch(`${AI_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(req.body)
        });

        if (response.ok) {
            const prediction = await response.json();
            res.json(prediction);
        } else {
            throw new Error(`AI 서버 응답 오류: ${response.status}`);
        }
    } catch (error) {
        console.error('AI 예측 요청 실패:', error);
        res.status(503).json({ 
            success: false, 
            error: 'AI 서버 연결 실패' 
        });
    }
});

// server.js (triggerModelRetraining 내부 수정)
async function triggerModelRetraining() {
    try {
        const recentData = await TrainingData.find({}).sort({ timestamp: -1 }).limit(1000).lean();
        const response = await fetch(AI_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(RETRAIN_SECRET ? { 'X-RETRAIN-SECRET': RETRAIN_SECRET } : {})
            },
            body: JSON.stringify({ trainingData: recentData })
        });
        console.log(response.ok ? '✅ AI 재훈련 완료' : `⚠️ 재훈련 실패: ${response.status}`);
    } catch (e) {
        console.error('❌ AI 재훈련 호출 오류:', e);
    }
}

// ===== 정적서빙 (public/index.html) =====
app.use(express.static(path.join(__dirname, '..', 'public')));

// ===== 서버 시작 =====
app.listen(PORT, () => {
    console.log(`🚀 Web/API on http://localhost:${PORT}`);
});
