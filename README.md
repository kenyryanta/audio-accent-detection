# 🧠 Wav2Vec 2.0 Accent Classification: Fine-Tuning Pipeline

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow?style=for-the-badge)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

## 📑 Executive Summary
This repository contains the **Model Training & Evaluation** logic for an Accent Recognition System. Following the data preparation phase, this pipeline fine-tunes a pre-trained **Wav2Vec 2.0 Base** model to discriminate between three distinct English accents: **American (US)**, **British (England)**, and **Australian (AU)**.

By leveraging **Transfer Learning**, we project the high-dimensional acoustic features learned from massive unlabeled datasets (Librispeech) onto our specific classification task, achieving robust performance even with a balanced subset of the Common Voice dataset.

## 🏗️ Model Architecture

The system utilizes the `Wav2Vec2ForSequenceClassification` architecture:

1.  **Feature Encoder (CNN)**:
    * Raw audio waveform ($16kHz$) $\rightarrow$ Latent speech representations.
    * Consists of 7 convolutional layers with temporal downsampling.
2.  **Context Network (Transformer)**:
    * 12 Transformer blocks with self-attention mechanisms.
    * Captures long-range dependencies and phonetic context.
3.  **Classification Head**:
    * A projection layer that maps the hidden states to the target class logits ($N=3$).
    * Applies Mean Pooling over the time dimension before the final linear layer.

## ⚙️ Prerequisites & Environment

### Hardware Requirements
* **GPU**: NVIDIA Tesla T4, P100, or V100 (Recommended 16GB+ VRAM).
* **RAM**: 12GB+ System RAM.

### Software Dependencies
Ensure you have the processed dataset (DatasetDict) from the previous pipeline stage.

```bash
pip install torch torchaudio torchvision
pip install transformers datasets evaluate accelerate scikit-learn
