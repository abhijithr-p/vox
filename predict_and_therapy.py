import argparse
import joblib
import os
import sounddevice as sd
import soundfile as sf
from data_utils import extract_features
from dotenv import load_dotenv
import textwrap
import numpy as np

# Optional: speech recognition for transcription
try:
    import speech_recognition as sr
except ImportError:
    sr = None

load_dotenv()

MODEL_PATH = "models/rf_model.joblib"
CLASSES_PATH = "models/classes.joblib"


# 🎙 Record live audio
def record_realtime(duration=5, samplerate=16000, filename="temp.wav"):
    """Record live voice from microphone"""
    print("🎙 Recording live... Speak now.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Saved {filename}")
    return filename


# 🧠 Load trained model and classes
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(
            "❌ Model files missing! Please train the model first using train_ml.py"
        )
    clf = joblib.load(MODEL_PATH)
    classes = joblib.load(CLASSES_PATH)
    return clf, classes


# 🩺 Therapy feedback generator
def generate_therapy_text(label_name, confidence=0.0):
    """Rule-based therapy feedback text."""
    if label_name.lower() == "fluent":
        return textwrap.dedent(f"""
            ✅ No major disorder detected (confidence {confidence:.2f}). 
            • Practice reading aloud daily for 5–10 minutes.
            • Maintain steady pace and clear articulation.
            • Try deep breathing exercises before speaking.
        """)
    elif label_name.lower() == "stutter":
        return textwrap.dedent(f"""
            ⚠️ Possible stuttering detected (confidence {confidence:.2f}).
            • Practice slow, deliberate speech: one sentence at a time.
            • Try vowel prolongation (stretching vowel sounds).
            • Relax shoulders and jaw to reduce tension.
        """)
    elif label_name.lower() == "dysarthria":
        return textwrap.dedent(f"""
            ⚠️ Possible motor-speech issue (dysarthria) detected (confidence {confidence:.2f}).
            • Focus on exaggerated articulation.
            • Practice consonant clusters like 'pa-ta-ka' slowly.
            • Strengthen mouth muscles with lip and tongue exercises.
        """)
    else:
        return textwrap.dedent(f"""
            ⚠️ Possible speech difficulty (confidence {confidence:.2f}).
            • Speak slowly and clearly.
            • Practice reading paragraphs daily.
            • If issues persist, consult a certified speech therapist.
        """)


# 🧠 Predict speech disorder + generate feedback
def predict_and_reply(audio_path, do_transcribe=False):
    clf, classes = load_model()
    feat = extract_features(audio_path)

    # ✅ Ensure consistent feature size
    expected_features = clf.n_features_in_
    if len(feat) != expected_features:
        print(f"⚠️ Adjusting feature length: got {len(feat)}, expected {expected_features}")
        feat = (
            feat[:expected_features]
            if len(feat) > expected_features
            else np.pad(feat, (0, expected_features - len(feat)), mode="constant")
        )

    # 🔍 Predict
    probs = clf.predict_proba([feat])[0]
    idx = int(probs.argmax())
    label = classes[idx]
    confidence = float(probs[idx])
    therapy = generate_therapy_text(label, confidence)

    # 🗣️ Optional: speech-to-text transcription
    transcript = None
    if do_transcribe and sr is not None:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio)
        except Exception:
            transcript = "(Unable to transcribe audio)"

    result = {
        "label": label,
        "confidence": confidence,
        "therapy": therapy,
    }
    if transcript:
        result["transcript"] = transcript

    return result


# 🧾 Command-line entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🎙 VocalEase AI — Voice Disorder Prediction")
    parser.add_argument("audio", nargs="?", help="Path to WAV audio file (optional — if not given, record live)")
    parser.add_argument("--transcribe", action="store_true", help="Enable speech-to-text transcription")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration (seconds) if live")
    args = parser.parse_args()

    if args.audio:
        audio_path = args.audio
    else:
        audio_path = record_realtime(duration=args.duration)

    result = predict_and_reply(audio_path, do_transcribe=args.transcribe)

    print("\n🧠 Predicted Disorder:", result["label"])
    print("Confidence:", round(result["confidence"] * 100, 2), "%")
    if "transcript" in result:
        print("\n🗣️ Transcript:", result["transcript"])
    print("\n🩺 Therapy Suggestion:\n", result["therapy"])
