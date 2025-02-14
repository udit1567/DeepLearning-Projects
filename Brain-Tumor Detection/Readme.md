# Brain Tumor Classification using CNN

## Overview
This project implements a deep learning model to classify brain tumors into four categories: 
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

The project uses a Convolutional Neural Network (CNN) for image classification and a Flask web application to provide an interface for uploading MRI images and viewing predictions.

## Features
- **Trained CNN Model**: A deep learning model built with PyTorch for brain tumor classification.
- **Flask Web App**: A user-friendly interface to upload MRI scans and get predictions.
- **Histogram Visualization**: Displays classification probabilities for better interpretability.
- **GPU Support**: Utilizes CUDA for faster inference when available.

## Dataset
The model is trained on a dataset of MRI brain scans categorized into four classes. Images are preprocessed and normalized before being fed into the CNN model.

## Installation
### Prerequisites
Ensure you have Python installed (preferably Python 3.8 or later). Install the required dependencies using:
```sh
pip install -r requirements.txt
```

### Required Libraries
- Flask
- PyTorch
- torchvision
- PIL (Pillow)
- Matplotlib
- NumPy

## Model Architecture
The CNN model consists of:
- **Convolutional Layers**: Extract spatial features from the MRI images.
- **ReLU Activation**: Introduces non-linearity for better feature learning.
- **Max Pooling Layers**: Reduces spatial dimensions while preserving important features.
- **Fully Connected Layers**: Maps extracted features to classification labels.
- **Softmax Activation**: Outputs probability distribution over classes.

## How to Run
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/brain-tumor-classification.git
   cd brain-tumor-classification
   ```
2. Run the Flask application:
   ```sh
   python app.py
   ```
3. Open a browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```
4. Upload an MRI image to classify.

## Model Inference
- The uploaded image is resized to `224x224`.
- It is normalized before passing through the trained CNN model.
- The model outputs class probabilities.
- The highest probability determines the predicted class.
- A histogram is generated to visualize classification confidence.

## Example Outputs
Upon uploading an MRI scan, the web app will display:
- The **Predicted Class** (e.g., `Glioma`)
- **Confidence Score** (e.g., `95.2%`)
- **Processing Time**
- **Histogram Visualization**
- **Description of the Predicted Tumor Type**

## Future Improvements
- Enhance the CNN architecture for higher accuracy.
- Implement Grad-CAM for visualizing model decision-making.
- Deploy as a cloud-based service.
- Add support for additional brain tumor types.

## Acknowledgments
The dataset and medical insights were sourced from publicly available MRI datasets.

## License

