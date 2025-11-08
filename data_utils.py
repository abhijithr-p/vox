# backend/data_utils.py
import librosa
import numpy as np

def extract_features(file_path):
    """
    Extract consistent numerical features (MFCC, Chroma, Spectral Contrast)
    from an audio file. Returns a fixed-length 60-dimensional feature vector.
    """
    try:
        # 1️⃣ Load audio (mono, fixed sampling rate)
        y, sr = librosa.load(file_path, sr=16000, mono=True)

        # 2️⃣ Extract key audio features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)                # tone, rhythm
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)                  # pitch, harmony
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)     # clarity, energy

        # 3️⃣ Combine them into a single numeric array
        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spec_contrast, axis=1)
        ])

        # 4️⃣ Standardize the feature length to 60
        target_len = 60
        if len(features) < target_len:
            features = np.pad(features, (0, target_len - len(features)), mode="constant")
        elif len(features) > target_len:
            features = features[:target_len]

        return features

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        # In case of failure, return a neutral zero-vector
        return np.zeros(60)


def extract_features_batch(file_paths):
    """
    Batch extractor for multiple audio files.
    Converts each .wav file path into a consistent feature vector.
    Returns a 2D NumPy array of shape (n_files, 60).
    """
    all_features = []
    for path in file_paths:
        feat = extract_features(path)
        all_features.append(feat)
    return np.array(all_features)
