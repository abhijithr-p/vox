import json
import time
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to record speech
def record_speech(duration=5, samplerate=16000, filename="temp.wav"):
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, audio_data, samplerate)
    return filename, duration

# Function to process recorded speech
def process_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"

# Fluency analysis
def analyze_fluency(text, duration):
    words = text.split()
    num_words = len(words)
    words_per_second = num_words / duration if duration > 0 else 0
    fluency_score = min(10, max(2, words_per_second * 5))  # Adjusted for realism
    return num_words, round(fluency_score, 2)

# Get recommended exercises
def get_exercise_recommendations(fluency_score):
    if fluency_score >= 8:
        return "✅ Great job! Keep practicing daily conversations."
    elif fluency_score >= 5:
        return "🗣 Try slow reading aloud & tongue twisters."
    else:
        return "🔄 Practice breathing exercises & simple sentence repetition."

# Disorder detection
def detect_disorder(text, fluency_score):
    disorder_message = "⚠️ No major speech disorder detected but slightly have a fluency issue."
    
    if fluency_score < 4:
        disorder_message = "❌ Possible speech disorder: Low fluency."
    elif any(word in text.lower() for word in ["uh", "um", "err", "ahh"]):
        disorder_message = "⚠️ Frequent stumbling detected."

    exercise = get_exercise_recommendations(fluency_score)
    return disorder_message, exercise

# API Route to record speech and process it
@app.get("/record")
async def record_and_analyze():
    audio_file, duration = record_speech()
    text = process_audio(audio_file)

    if text:
        num_words, fluency = analyze_fluency(text, duration)
        disorder_status, exercise = detect_disorder(text, fluency)

        return JSONResponse({
            "transcribed_text": text,
            "duration": round(duration, 2),
            "words_spoken": num_words,
            "fluency_score": fluency,
            "disorder_analysis": disorder_status,
            "exercise": exercise
        })
    return JSONResponse({"error": "Could not recognize speech."}, status_code=400)

# API to process uploaded audio
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    file_path = f"upload/{file.filename}"
    os.makedirs("upload", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = process_audio(file_path)

    if text:
        num_words, fluency = analyze_fluency(text, 5)  # Assuming a 5-second duration
        disorder_status, exercise = detect_disorder(text, fluency)

        return JSONResponse({
            "transcribed_text": text,
            "words_spoken": num_words,
            "fluency_score": fluency,
            "disorder_analysis": disorder_status,
            "exercise": exercise
        })
    return JSONResponse({"error": "Could not process the file."}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
