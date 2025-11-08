# backend/main.py
import os
import sys
import tempfile
import traceback
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from data_utils import extract_features

# === Paths ===
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.joblib")

# === FastAPI Setup ===
app = FastAPI(title="VocalEase AI", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clf = None
classes = None

# ============================================================
# 🧠 Auto-create model if not found
# ============================================================
def ensure_model_exists():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        print("⚠️ Model not found — creating a dummy one...")
        classes = ["fluent", "stutter", "dysarthria"]
        X = np.random.rand(100, 60)
        y = np.random.choice(classes, 100)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)

        joblib.dump(clf, MODEL_PATH)
        joblib.dump(classes, CLASSES_PATH)
        print("✅ Dummy model created and saved!")
    else:
        print("✅ Model files already exist.")


# ============================================================
# 🚀 Load Model on Startup
# ============================================================
@app.on_event("startup")
async def load_model_on_startup():
    global clf, classes
    ensure_model_exists()
    try:
        clf = joblib.load(MODEL_PATH)
        classes = joblib.load(CLASSES_PATH)
        print(f"✅ Model loaded successfully with classes: {classes}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)


# ============================================================
# 🎯 Therapy Text Generator
# ============================================================
def generate_therapy_text(label_name, confidence=0.0):
    if label_name.lower() == "fluent":
        return f"✅ No major disorder detected ({confidence:.2f}).\nPractice reading aloud daily."
    elif label_name.lower() == "stutter":
        return f"⚠️ Possible stuttering ({confidence:.2f}).\nSpeak slowly and stay relaxed."
    elif label_name.lower() == "dysarthria":
        return f"⚠️ Possible motor-speech issue ({confidence:.2f}).\nFocus on articulation exercises."
    else:
        return f"⚠️ Possible speech issue ({confidence:.2f}).\nConsider consulting a speech therapist."


# ============================================================
# 🩺 Analyze Audio API
# ============================================================
@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    tmp_path = None
    try:
        if not file.filename.endswith(".wav"):
            return JSONResponse(status_code=400, content={"error": "Only .wav files are supported."})

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        feat = extract_features(tmp_path)
        probs = clf.predict_proba([feat])[0]
        idx = int(probs.argmax())
        label = classes[idx]
        confidence = float(probs[idx])
        therapy = generate_therapy_text(label, confidence)

        return {"label": label, "confidence": confidence, "therapy": therapy}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# 💚 Health Check
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok", "message": "VocalEase AI Server is running"}


# ============================================================
# ▶️ Run Server
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n🎤 Starting VocalEase AI Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
