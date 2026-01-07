# Sleep Apnea Detection using ECG Signal (CNN–LSTM)

This repository presents a deep learning–based system for **automatic detection of Obstructive Sleep Apnea (OSA)** using single-lead ECG signals.  
The proposed approach combines **1D Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks to learn both spatial and temporal patterns from ECG-derived features.

The model is evaluated on the **PhysioNet Apnea-ECG dataset** and achieves strong performance while maintaining low computational complexity, making it suitable for real-world and remote healthcare applications.

---

## 🔬 Problem Motivation

Obstructive Sleep Apnea (OSA) is a common but underdiagnosed sleep disorder that can lead to serious cardiovascular and neurological complications.  
Traditional diagnosis using **Polysomnography (PSG)** is expensive, complex, and time-consuming.

ECG-based automated detection provides a **cost-effective and scalable alternative**, especially suitable for wearable and remote monitoring systems.

---

## 🧠 Proposed Architecture Overview

The system follows a **feature-based deep learning pipeline**, where handcrafted ECG features are fed into a CNN–LSTM model for classification.

### 🔷 High-Level Pipeline

ECG Signal
↓
Preprocessing
↓
Segmentation
↓
Feature Extraction
↓
Feature Scaling & Balancing
↓
CNN Feature Learning
↓
LSTM Temporal Modeling
↓
Softmax Classifier
(Apnea / Non-Apnea)


---

## 📊 Feature Set Description (40 Features)

### ⏱️ Time-Domain (19)

### 📈 Frequency-Domain (10)

### 🔄 Non-Linear (11)

---

## ⚙️ Training Strategy

- **10-Fold Cross-Validation**
- **SMOTE** for class imbalance handling
- **MinMax normalization**
- **L2 regularization + Dropout** to prevent overfitting
- **Early stopping + ReduceLROnPlateau**

---

## 📈 Performance (PhysioNet Apnea-ECG)

| Metric | Value |
|------|------|
| Accuracy | **89.14%** |
| Sensitivity | **91.79%** |
| Specificity | **86.49%** |
| F1-score | **89.42%** |
| AUC-ROC | **0.9553** |

---

## 🧪 Repository Structure

Sleep_Apnea_Detection_using_ECG_Signal_CNN-LSTM_model/
│
├── src/
│ ├── preprocessing/
│ ├── training/
│ ├── inference/
│
├── artifacts/
│ ├── cnn_lstm_apneamodel.keras
│ ├── mean_imputer.pkl
│ ├── minmax_scaler.pkl
│ └── feature_order.pkl
│
├── data/
├── README.md
├── requirements.txt


---

## 🚀 Deployment Ready

The trained model and preprocessing artifacts are saved separately and can be directly used in:
- **FastAPI REST service**
- **Docker container**
- **Cloud deployment (AWS / GCP / Azure)**
- **Remote health monitoring systems**


⭐ If you find this work useful, please consider starring the repository also cite this paper [https://doi.org/10.1007/978-3-032-11335-1_15]

