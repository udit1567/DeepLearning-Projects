import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
import io
import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


app = Flask(__name__)

# Define classes
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Paragraph descriptions for each class
CLASS_INFO = {
    'glioma': '''Overview: Meningiomas arise from the meninges, the protective layers surrounding the brain and spinal cord. Most are benign but can still cause significant issues due to their size or location.
    Grades:
    Grade I: Benign and slow-growing.
    Grade II (Atypical): Faster-growing with a higher recurrence rate.
    Grade III (Malignant): Aggressive and more likely to invade surrounding tissues.
    Symptoms: Vision changes, headaches, memory loss, seizures, or weakness depending on location.
    Treatment: Observation for small tumors; surgery or radiation for larger or symptomatic ones.''',
    'meningioma': '''Overview: Meningiomas arise from the meninges, the protective layers surrounding the brain and spinal cord. Most are benign but can still cause significant issues due to their size or location.
    Grades:
    Grade I: Benign and slow-growing.
    Grade II (Atypical): Faster-growing with a higher recurrence rate.
    Grade III (Malignant): Aggressive and more likely to invade surrounding tissues.
    Symptoms: Vision changes, headaches, memory loss, seizures, or weakness depending on location.
    Treatment: Observation for small tumors; surgery or radiation for larger or symptomatic ones.''',
    'notumor': "No tumor detected in this scan. The brain appears normal based on the provided image.",
    'pituitary': '''Overview: These tumors develop in the pituitary gland, a small organ at the base of the brain responsible for hormone production. Most pituitary tumors are benign (non-cancerous) and are called pituitary adenomas.
    Types:
    Functioning: These produce excess hormones, leading to symptoms like Cushing's disease or acromegaly.
    Non-functioning: These do not produce hormones but may cause symptoms due to their size, such as headaches or vision problems.
    Treatment: Surgery, medication to control hormone levels, and sometimes radiation therapy.'''
}

# Load trained model
class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding="same"), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding="same"), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256* 28 * 28, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(input_channels=3).to(device)

# Load trained weights
model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=device))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to predict class
def predict_class(image):
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    probabilities = torch.softmax(output, dim=1)[0]
    predicted_class = torch.argmax(probabilities).item()
    return CLASS_NAMES[predicted_class], probabilities.cpu().detach().numpy()

# Generate histogram
def generate_histogram(probabilities):
    plt.figure(figsize=(6, 4))
    plt.bar(CLASS_NAMES, probabilities, color=['red', 'blue', 'green', 'purple'])
    plt.xlabel("Brain Tumor Type")
    plt.ylabel("Probability")
    plt.title("Classification Probabilities")
    plt.ylim([0, 1])
    plt.savefig("static/histogram.png")
    plt.close()

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        start_time = time.time()
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image.save("static/uploaded_image.jpg")  # Save image for display
        predicted_class, probabilities = predict_class(image)
        confidence = round(np.max(probabilities) * 100, 2)
        processing_time = round(time.time() - start_time, 2)
        generate_histogram(probabilities)
        
        return render_template("result.html",confidence=confidence,
                               processing_time=processing_time,
                               image="static/uploaded_image.jpg",
                               histogram="static/histogram.png",
                               result=predicted_class,
                               description=CLASS_INFO[predicted_class])

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
