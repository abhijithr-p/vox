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
app = FastAPI(
    title="VoxThera AI",
    description="AI-based speech disorder detection and therapy recommendation API",
    version="1.0.0",
)

# ✅ CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace * with ["https://your-frontend-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clf = None
classes = None

# ============================================================
# 🧠 Ensure Model Exists
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
        return (
            f"✅ No major disorder detected (confidence {confidence:.2f}).\n"
            f"• Practice reading aloud daily for 5–10 minutes.\n"
            f"• Maintain steady pace and clear articulation.\n"
            f"• Try deep breathing before speaking."
        )
    elif label_name.lower() == "stutter":
        return (
            f"⚠️ Possible stuttering detected (confidence {confidence:.2f}).\n"
            f"• Practice slow, deliberate speech.\n"
            f"• Try vowel prolongation and stay relaxed.\n"
            f"• Relax your shoulders and jaw to reduce tension."
        )
    elif label_name.lower() == "dysarthria":
        return (
            f"⚠️ Possible motor-speech issue (dysarthria) detected (confidence {confidence:.2f}).\n"
            f"• Focus on exaggerated articulation.\n"
            f"• Practice consonant clusters slowly.\n"
            f"• Strengthen mouth muscles with lip and tongue exercises."
        )
    else:
        return (
            f"⚠️ Possible speech difficulty detected (confidence {confidence:.2f}).\n"
            f"• Speak slowly and clearly.\n"
            f"• Practice reading paragraphs daily.\n"
            f"• If issues persist, consult a speech therapist."
        )


# ============================================================
# 🩺 Analyze Audio API
# ============================================================
@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    tmp_path = None
    try:
        # Validate file type
        if not file.filename.endswith(".wav"):
            return JSONResponse(
                status_code=400,
                content={"error": "Only .wav files are supported."},
            )

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extract features
        feat = extract_features(tmp_path)

        # Predict
        probs = clf.predict_proba([feat])[0]
        idx = int(probs.argmax())
        label = classes[idx]
        confidence = float(probs[idx])
        therapy = generate_therapy_text(label, confidence)

        print(f"✅ Sending response: {label}, {confidence:.2f}")
        return {"label": label, "confidence": confidence, "therapy": therapy}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# 💚 Health Check + Root Route
# ============================================================
@app.get("/")
async def root():
    return {"message": "🎤 VoxThera AI Backend is running successfully!"}

@app.get("/health")
async def health():
    return {"status": "ok", "message": "VoxThera AI server healthy"}


# ============================================================
# ▶️ Local Run (for debugging)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n🎤 Starting VoxThera AI Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
