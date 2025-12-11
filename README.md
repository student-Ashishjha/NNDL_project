🔍 High-Accuracy Real vs Fake Face Classification Using Deep Learning
<p align="center"> <img src="https://github.com/user-attachments/assets/detective-ai.gif" width="250"> </p> <p align="center"> <b>⚡ A powerful AI system that detects Deepfake facial images with up to 99% accuracy.</b><br> Built using a Custom Convolutional Neural Network trained on FFHQ (Real) and TPDNE (Fake) datasets. </p>
Deepfakes have become one of the biggest digital threats in today’s world.
This project builds a Deepfake Detection System using a Custom CNN capable of identifying Real vs AI-generated faces with exceptional accuracy.

The model learns subtle manipulation patterns, unnatural textures, and inconsistencies in fake images that are invisible to the human eye.





✔ Achieves 98–99% validation accuracy
✔ Trained on 6000 high-quality images
✔ Custom CNN architecture — no transfer learning used
✔ Includes evaluation metrics, ROC curve, and confusion matrix
✔ User-friendly prediction module

Project Architecture

             ┌────────────────────┐
             │  Dataset Loading   │
             └────────┬───────────┘
                      │
             ┌────────▼───────────┐
             │  Image Preprocess  │
             └────────┬───────────┘
                      │
             ┌────────▼───────────┐
             │   CNN Model Build  │
             └────────┬───────────┘
                      │
             ┌────────▼───────────┐
             │   Model Training   │
             └────────┬───────────┘
                      │
             ┌────────▼───────────┐
             │ Model Evaluation   │
             └────────┬───────────┘
                      │
             ┌────────▼───────────┐
             │   Prediction App    │
             └─────────────────────┘
Dataset Used:
| Dataset Name                       | Type | Count |
| ---------------------------------- | ---- | ----- |
| **FFHQ Face Dataset**              | Real | 3,000 |
| **ThisPersonDoesNotExist (TPDNE)** | Fake | 3,000 |
Total images used: 6,000
Split: 80% Train — 20% Validation

Image Preprocessing

✔ Resize to 128 × 128 × 3
✔ Convert to NumPy array
✔ Normalize pixel values to 0–1
✔ Encode labels:

0 → Real

1 → Fake

Model Architecture (Custom CNN)
🔹 Convolutional Layers

Filters: 64 → 32 → 16

ReLU activation

Dilated Convolution for wider context

🔹 Pooling Layers

MaxPooling2D for feature downsampling

🔹 Dense Layers

400 → 512 → 400 neurons

Dropout (0.5) for regularization

🔹 Output Layer

Dense(2) + Sigmoid activation

🔹 Compilation
loss = 'binary_crossentropy'
optimizer = Adam(learning_rate=1e-5)
metrics = ['accuracy']


Training Summary

Epochs: 100

Batch Size: 32

Callback: ModelCheckpoint (saves best model automatically)

Performance Visualization:

Accuracy curves

Loss curves
    | Metric        | Score     |
    | ------------- | --------- |
    | **Accuracy**  | 99%       |
    | **Precision** | 99.17%    |
    | **Recall**    | 98.85%    |
    | **Loss**      | Very Low  |
    | **AUC**       | Excellent |

Prediction Module

✔ Upload any face image
✔ Automatically preprocess
✔ Model predicts:Real Face   OR   Fake (AI-Generated)

Hardware Requirements

CPU — supported (slow)

💡 GPU recommended (NVIDIA Tesla T4 / P100 / V100)

Implemented on Kaggle GPU Environment

🏁 Conclusion

The Deepfake Detection System proves the capability of Custom CNNs to accurately identify manipulated facial images.
It serves as a powerful tool for digital forensics, security agencies, and social media verification pipelines.

