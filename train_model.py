import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

# fungsi ekstraksi fitur suara
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    jitter = np.std(pitch) / np.mean(pitch) if np.mean(pitch) > 0 else 0
    shimmer = np.std(y) / np.mean(np.abs(y))
    features = np.hstack([mfccs, np.mean(pitch), jitter, shimmer])
    return features

# folder dataset
data_folder = "data/"

# siapkan list X dan y
X = []
y = []

# baca semua file di subfolder jenuh (label = 1)
jenuh_folder = os.path.join(data_folder, "jenuh")
for file_name in os.listdir(jenuh_folder):
    if file_name.endswith(".wav") or file_name.endswith(".mp3"):
        file_path = os.path.join(jenuh_folder, file_name)
        features = extract_features(file_path)
        X.append(features)
        y.append(1)

# baca semua file di subfolder tidak_jenuh (label = 0)
tidak_jenuh_folder = os.path.join(data_folder, "tidak_jenuh")
for file_name in os.listdir(tidak_jenuh_folder):
    if file_name.endswith(".wav") or file_name.endswith(".mp3"):
        file_path = os.path.join(tidak_jenuh_folder, file_name)
        features = extract_features(file_path)
        X.append(features)
        y.append(0)

# cek distribusi dataset
print("Jumlah data:", len(y))
print("Distribusi label:", {0: y.count(0), 1: y.count(1)})

# latih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# simpan model ke file
joblib.dump(model, "model.pkl")
print("✅ Model sudah dilatih dengan dataset folder terpisah & disimpan sebagai model.pkl")
