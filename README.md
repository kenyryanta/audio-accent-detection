# 🤖 Wav2Vec2 Accent Classifier - Training Pipeline

Repository ini berisi kode untuk melatih (fine-tuning) model **Wav2Vec 2.0** agar dapat mengklasifikasikan aksen bahasa Inggris: **United States (US)**, **England (UK)**, dan **Australia (AU)**.

Ini adalah **Tahap 2** dari proyek, yang dijalankan setelah *Data Preparation*.

## 📋 Daftar Isi
- [Overview Model](#-overview-model)
- [Prasyarat](#-prasyarat)
- [Arsitektur Training](#-arsitektur-training)
- [Hyperparameters](#-hyperparameters)
- [Cara Menjalankan Training](#-cara-menjalankan-training)
- [Evaluasi & Metrik](#-evaluasi--metrik)
- [Hasil yang Diharapkan](#-hasil-yang-diharapkan)

## 🧠 Overview Model
Kami menggunakan pendekatan **Transfer Learning**:
* **Base Model**: `facebook/wav2vec2-base` (Pre-trained pada 960 jam audio Librispeech).
* **Task**: *Audio Classification* (Sequence Classification).
* **Input**: Raw Audio Waveform (16kHz).
* **Output**: Probabilitas untuk 3 kelas aksen.

## 🛠️ Prasyarat
Sebelum menjalankan `code_nomor_2.ipynb`, pastikan Anda telah menyelesaikan tahap persiapan data (Code Nomor 1) dan memiliki file dataset yang sudah diproses.

### Library yang Dibutuhkan
```bash
pip install transformers datasets evaluate torch torchaudio accelerate scikit-learn
