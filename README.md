# Obstructive Sleep Apnea Detection using ECG Signals with a 1D CNN-LSTM Model

Obstructive Sleep Apnea (OSA) is a prevalent sleep disorder that can lead to serious health complications if left undiagnosed. Conventional diagnosis using Polysomnography (PSG) is complex, expensive, and time-consuming, limiting its accessibility for large-scale screening.

This repository presents an automatic OSA detection framework based on electrocardiogram (ECG) signals using a one-dimensional Convolutional Neural Network combined with Long Short-Term Memory (1D CNN-LSTM) architecture. ECG signals provide critical cardiac information closely linked to respiratory dynamics during sleep, making them suitable for non-intrusive apnea detection.

The ECG recordings are preprocessed using filtering and normalization techniques to enhance signal quality. A comprehensive set of 41 handcrafted features spanning time-domain, frequency-domain, and non-linear domain is extracted to capture physiological variations associated with apnea events. The CNN layers learn discriminative spatial features, while the LSTM layers model temporal dependencies essential for accurate classification.

The proposed model is evaluated on the PhysioNet Apnea-ECG dataset using 10-fold cross-validation, achieving strong performance with 89.14% accuracy, 91.79% sensitivity, 86.49% specificity, 89.42% F1-score, and an AUC-ROC of 0.9553, demonstrating robust generalization with minimal overfitting.

Owing to its high performance and low computational complexity, this approach is well-suited for real-world OSA screening applications and provides a scalable foundation for future clinical and deployment-oriented enhancements.
## Methodology

1. ECG Preprocessing
   - 50 Hz notch filtering
   - 0.5 Hz high-pass filtering
   - Z-score normalization
   - 1-minute segmentation

2. Feature Extraction
   - Time-domain HRV features
   - Frequency-domain features
   - Non-linear features

3. Classification
   - 1D CNN for spatial feature learning
   - LSTM for temporal dependency modeling
   - Softmax output for binary classification
