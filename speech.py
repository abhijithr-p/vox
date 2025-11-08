import json
import time
import wave
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import argparse
import os

def record_speech(duration=5, samplerate=16000, filename="temp.wav"):
    print("\U0001F3A7 Speak now...")
    
    # Record audio
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait for recording to complete
    
    # Save the recorded audio
    sf.write(filename, audio_data, samplerate)
    
    return filename, duration

def process_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"

def analyze_fluency(text, duration):
    words = text.split()
    num_words = len(words)
    words_per_second = num_words / duration if duration > 0 else 0
    
    if words_per_second > 2:
        fluency_score = 10
    elif words_per_second > 1:
        fluency_score = 7
    elif words_per_second > 0.5:
        fluency_score = 5
    else:
        fluency_score = 2
    
    return num_words, fluency_score

def detect_disorder(text, fluency_score):
    """Detects speech disorders based on fluency and clarity."""
    words = text.split()
    if len(words) < 3:
        return "❌ Possible speech disorder detected: Speech too short."
    elif any(word in text.lower() for word in ["uh", "um", "err", "ahh"]):
        return "⚠️ Possible speech disorder: Frequent stumbling detected."
    elif len(max(words, key=len, default="")) > 15:
        return "⚠️ Possible speech disorder: Unusually long words detected."
    elif fluency_score < 5:
        return "⚠️ Possible speech disorder: Low fluency detected."
    elif fluency_score < 7:
        return "⚠️ Possible speech disorder: Slight fluency issues detected."
    return "✅ No major speech disorder detected."

def provide_feedback(fluency_score):
    if fluency_score == 10:
        return "🌟 Excellent fluency! Your speech is clear and well-paced."
    elif fluency_score >= 7:
        return "👍 Good fluency, but slight improvements can be made."
    elif fluency_score >= 5:
        return "⚠️ Moderate fluency issues. Try speaking more smoothly."
    else:
        return "❌ Significant fluency issues detected. Consider speech exercises."

def suggest_exercises(fluency_score, disorder_status):
    """Suggests speech exercises based on detected disorder and fluency score."""
    exercises = []
    
    if "Frequent stumbling" in disorder_status:
        exercises.append("🎤 Try reading aloud daily to improve articulation.")
        exercises.append("🎵 Practice tongue twisters like 'She sells seashells by the seashore.'")
    
    if "Speech too short" in disorder_status:
        exercises.append("📢 Practice expanding your sentences. Try describing an object in five sentences.")
    
    if "Low fluency detected" in disorder_status or fluency_score < 5:
        exercises.append("🗣️ Slow down your speech and focus on pronunciation.")
        exercises.append("🎶 Use rhythmic speaking exercises (e.g., speaking in a musical rhythm).")
        exercises.append("🔄 Repeat phrases slowly and then gradually speed up.")
    
    if "Unusually long words detected" in disorder_status:
        exercises.append("📚 Break down long words into syllables and pronounce each part clearly.")
    
    if not exercises:
        exercises.append("✅ Keep practicing! Try recording yourself and analyzing your speech.")

    return exercises

def convert_m4a_to_wav(m4a_path):
    wav_path = m4a_path.replace(".m4a", ".wav")
    os.system(f"ffmpeg -i {m4a_path} {wav_path}") 
    return wav_path

def process_m4a_file(file_path):
    wav_path = convert_m4a_to_wav(file_path)
    return process_audio(wav_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Recognition & Fluency Analysis")
    parser.add_argument("--file", type=str, help="Path to M4A file (optional)")
    args = parser.parse_args()

    if args.file:
        print(f"\U0001F4C2 Processing file: {args.file}")
        text = process_m4a_file(args.file)
        duration = 1.5  # Assume default duration for file input
    else:
        audio_file, duration = record_speech()
        text = process_audio(audio_file)

    if text:
        num_words, fluency = analyze_fluency(text, duration)
        feedback = provide_feedback(fluency)
        disorder_status = detect_disorder(text, fluency)
        exercises = suggest_exercises(fluency, disorder_status)
        
        print(f"\U0001F4DD Transcribed Text: {text}")
        print(f"\U0001F551 Duration: {round(duration, 2)} sec")
        print(f"\U0001F5E3 Words Spoken: {num_words}")
        print(f"⚡ Fluency Score: {fluency} / 10")
        print(f"📝 Feedback: {feedback}")
        print(f"🏥 Disorder Analysis: {disorder_status}")
        
        print("\n🎯 Recommended Speech Exercises:")
        for exercise in exercises:
            print(f"- {exercise}")
    else:
        print("❌ Could not recognize speech.")
