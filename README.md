# 🧠 Deteksi Mental Fatigue Mahasiswa via Suara

Aplikasi ini menggunakan Machine Learning untuk mendeteksi kondisi mental fatigue (kelelahan mental) pada mahasiswa melalui analisis suara. Aplikasi dibangun dengan Python dan Streamlit.

## Fitur
- **Upload Audio**: Mendukung format WAV dan MP3.
- **Ekstraksi Fitur**: Menganalisis MFCC, Pitch, Jitter, dan Shimmer.
- **Prediksi Cerdas**: Mengklasifikasikan kondisi sebagai "Fatigue" atau "Non-Fatigue" beserta tingkat keyakinan (confidence).
- **Visualisasi**:
    - Grafik probabilitas prediksi.
    - Histogram distribusi pitch.
    - Radar chart profil fitur akustik.
- **Riwayat**: Menyimpan hasil analisis ke dalam `history.csv`.

## Prasyarat
Pastikan Anda telah menginstal Python di komputer Anda.

## Instalasi

1.  Clone atau download repository ini.
2.  Buka terminal atau command prompt di folder project.
3.  Instal library yang dibutuhkan dengan perintah:

    ```bash
    pip install -r requirements.txt
    ```

## Cara Menjalankan

Jalankan aplikasi dengan perintah berikut:

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser Anda (biasanya di `http://localhost:8501`).
