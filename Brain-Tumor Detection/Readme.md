# Brain Tumor Classification using CNN

## Project Overview
This project implements a **deep learning-based approach** for brain tumor classification using Convolutional Neural Networks (CNNs). The application is built using **Flask** to provide a web interface for users to upload MRI images and obtain predictions. The model classifies MRI scans into four categories:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary Tumor**

The project integrates **TensorBoard** for visualization, hyperparameter tuning, and performance tracking.

---

## Dataset
The dataset used for training and evaluation can be accessed from Kaggle:
🔗 **[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)**

The dataset consists of MRI scan images labeled according to the tumor type. Preprocessing steps include resizing images to **224x224**, normalization, and data augmentation.

---

## Network Architecture
The CNN model comprises **three convolutional layers**, each followed by **ReLU activation and max-pooling**. The extracted features are then passed through **fully connected layers** for classification.

### Model Structure:
- **Convolutional Layers:**
  - `Conv2D(64, kernel_size=3, padding="same")` → `ReLU()` → `MaxPool2D(2,2)`
  - `Conv2D(128, kernel_size=3, padding="same")` → `ReLU()` → `MaxPool2D(2,2)`
  - `Conv2D(256, kernel_size=3, padding="same")` → `ReLU()` → `MaxPool2D(2,2)`

- **Fully Connected Layers:**
  - Flatten layer
  - `Linear(256*28*28, 128)` → `ReLU()` → `Dropout(0.5)`
  - `Linear(128, 64)` → `ReLU()` → `Dropout(0.4)`
  - `Linear(64, 32)` → `ReLU()` → `Dropout(0.2)`
  - `Linear(32, 4)` (Output Layer with 4 classes)

- **Activation Functions:**
  - ReLU for feature extraction layer

---

## Installation & Setup
### Clone Repository
```
git clone https://github.com/udit1567/DeepLearning-Projects.git
cd DeepLearning-Projects/brain_tumor_classification
```

### Install Dependencies
Make sure you have Python installed (preferably Python 3.8+), then install the required libraries:
```
pip install -r requirements.txt
```

### Run Flask Application
```
python app.py
```
The application will be available at **http://127.0.0.1:5000/**.

---

## Features
✅ **Upload MRI Images** for classification
✅ **Displays Tumor Type with Confidence Score**
✅ **Generates Probability Histogram**
✅ **Provides Descriptive Information about Tumor Type**
✅ **Hyperparameter tuning with TensorBoard**

---

## Model Performance
📊 **Accuracy on Test Dataset:** 
![image](https://github.com/user-attachments/assets/8274518b-8c33-4573-90ec-ff6826fecebe)



📉 **LossAccuracy & 📈 Accuracy Curve:**
![Accuracy](https://github.com/user-attachments/assets/6a4fc2b0-ca32-48fc-bfa3-ffdc1edf641b)

---

## Screenshots
🚀 **Flask Web Interface:**
![Screenshot 2025-02-14 164247](https://github.com/user-attachments/assets/4da62f41-0ac3-43ab-b1ab-d3cedad5795e)


🎯 **Prediction Result:**
![Screenshot 2025-02-14 164259](https://github.com/user-attachments/assets/13b1f5ba-0003-47ed-a7f8-b4d40e9b6e56)


📊 **Probability Histogram:**

![histogram](https://github.com/user-attachments/assets/10df8132-81ae-414c-ab96-6fd98f7e77cb)



