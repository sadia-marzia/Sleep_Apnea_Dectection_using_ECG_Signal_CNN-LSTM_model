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

## Methodology

This repository implements the CNN–LSTM-based framework proposed in our Springer CCIS paper.
For a complete description of the signal preprocessing, feature extraction, model architecture, and evaluation protocol, please refer to the published article.

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


## 📄 Related Publication

This repository is associated with the following peer-reviewed publication:

**S. Marzia, M. J. Islam, R. Raen**,  
*Obstructive Sleep Apnea Detection Using 1D CNN-LSTM Approach*,  
International Conference on Data Science, AI and Applications (ICDSAIA 2025),  
Springer, *Communications in Computer and Information Science*, Vol. 2681.

**DOI:** https://doi.org/10.1007/978-3-032-11335-1_15

⚠️ *Note:* The published manuscript is not distributed in this repository due to publisher licensing restrictions.

---

## ⭐ Citation & Acknowledgment

If you find this work useful for your research or application, please consider:

⭐ Starring this repository

📖 Citing the associated paper
