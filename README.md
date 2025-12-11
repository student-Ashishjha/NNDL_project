# NNDL_project
1. Dataset

Two Kaggle datasets are used:

FFHQ Face Dataset → Real images

ThisPersonDoesNotExist 10k Dataset → AI-generated fake images

6,000 images (3,000 real + 3,000 fake) were used for training and validation.

🛠️ 2. Image Preprocessing

Images resized to 128 × 128 × 3

Converted to NumPy arrays

Pixel normalization to [0,1]

Labels:

0 → Real

1 → Fake

Dataset split:

80% Training

20% Validation

🧩 3. Model Architecture

A custom CNN (no transfer learning) is used.

🔹 Key Components:

Multiple Conv2D layers (filters: 64 → 32 → 16)

ReLU activation

Dilated convolutions

MaxPooling2D

GlobalAveragePooling2D

Dense layers (400 → 512 → 400)

Dropout (0.5)

Output layer: Dense(2) + Sigmoid

🛠 Model Compilation:

Loss: Binary Crossentropy

Optimizer: Adam (LR = 1e-5)

Metrics: Accuracy

🏋️ 4. Model Training

Training is performed using:

Batch size: 32

Epochs: 100

Validation tracking

ModelCheckpoint to save best weights

Visualizations:

Training vs Validation Accuracy

Training vs Validation Loss

📊 5. Model Evaluation

Evaluation metrics used:

Accuracy

Precision

Recall

Confusion Matrix

ROC Curve & AUC

The best model achieved:

Accuracy: ~99%

Precision: ~99.17%

Recall: ~98.85%

🔍 6. Prediction Module

To classify a new image:

Upload an image

Preprocess: resize → normalize → expand dims

Run through trained model

Output:

Real Face

Fake (AI-Generated) Face

Confidence score displayed

Example:

input_arr = img_to_array(load_img("image.png", target_size=(128,128,3))) / 255.0
input_arr = np.expand_dims(input_arr, axis=0)
prediction = np.argmax(model.predict(input_arr))

🖥️ 7. Hardware Requirements

Works on CPU

GPU recommended for training

Implementation done in Kaggle GPU Environment

📚 8. References

FaceForensics++ (Rossler et al., 2019)

MesoNet (Afchar et al., 2018)

Deepfake Detection Survey (Tolosana et al., 2020)

Capsule-Forensics (Nguyen et al., 2019)

TensorFlow Official Documentation

🎯 Conclusion

This project demonstrates how deep learning can effectively identify deepfake images by learning manipulation artifacts that are hard for humans to detect.
The system achieves high accuracy and can be extended to video-based deepfake detection, real-time inference, or integrated into security and verification systems.
