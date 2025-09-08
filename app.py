# app.py
import streamlit as st
import librosa
import numpy as np
import joblib
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# Load model yang sudah dilatih
model = joblib.load("model.pkl")

# fungsi ekstraksi fitur (sama kayak di train_model.py)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    jitter = np.std(pitch) / np.mean(pitch) if np.mean(pitch) > 0 else 0
    shimmer = np.std(y) / np.mean(np.abs(y))
    features = np.hstack([mfccs, np.mean(pitch), jitter, shimmer])
    return features

# fungsi simpan log ke CSV
def save_log(file_name, prediction, prob):
    log_file = "history.csv"

    new_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": file_name,
        "prediction": "Jenuh" if prediction == 1 else "Tidak Jenuh",
        "confidence": round(float(prob_jenuh if prediction == 1 else prob_tidak_jenuh), 2)
    }

    if not os.path.exists(log_file):
        df = pd.DataFrame([new_data])
        df.to_csv(log_file, index=False)
    else:
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_csv(log_file, index=False)

    return new_data

# UI Streamlit
st.title("🧠 Deteksi Kejenuhan Mental Mahasiswa")
st.write("Upload rekaman suara (WAV/MP3) untuk analisis kejenuhan.")

uploaded_file = st.file_uploader("Pilih file audio", type=["wav", "mp3"])

if uploaded_file is not None:
    # simpan file sementara
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # ekstrak fitur
    features = extract_features("temp.wav").reshape(1, -1)

    # prediksi
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    if len(prob) == 2:
        prob_tidak_jenuh = prob[0]
        prob_jenuh = prob[1]
    else:
        if prediction == 1:
            prob_tidak_jenuh = 0.0
            prob_jenuh = 1.0
        else:
            prob_tidak_jenuh = 1.0
            prob_jenuh = 0.0

    # tampilkan hasil prediksi
    if prediction == 1:
        st.error(f"⚠️ Mahasiswa terdeteksi **Jenuh** (Confidence {prob_jenuh*100:.2f}%)")
    else:
        st.success(f"✅ Mahasiswa terdeteksi **Tidak Jenuh** (Confidence {prob_tidak_jenuh*100:.2f}%)")

    # tampilkan grafik probabilitas
    fig, ax = plt.subplots()
    labels = ["Tidak Jenuh", "Jenuh"]
    ax.bar(labels, [prob_tidak_jenuh, prob_jenuh], color=["green", "red"])
    ax.set_ylabel("Probabilitas")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # simpan hasil ke CSV
    save_log(uploaded_file.name, prediction, prob)
    st.write("📂 Hasil tersimpan ke **history.csv**")

# tampilkan tabel riwayat & tombol clear history
if os.path.exists("history.csv"):
    st.subheader("Riwayat Analisis")
    df_history = pd.read_csv("history.csv")
    st.dataframe(df_history)

    # tombol untuk hapus riwayat
    if st.button("🗑️ Hapus Riwayat"):
        os.remove("history.csv")
        st.warning("Riwayat berhasil dihapus! Silakan refresh halaman.")
