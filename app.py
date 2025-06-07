import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from io import BytesIO

# Feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.features.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, 1)
        return x

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def extract_features(image_paths, model, device, transform):
    features = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        feat = model(img_tensor).cpu().numpy()[0]
        features.append(feat)
    return np.array(features)

def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))

# Load and cache model and train data
@st.cache_resource
def load_model_and_embeddings(train_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor().to(device)
    transform = get_transform()

    train_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)
                   if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    st.write(f"Loaded {len(train_paths)} training images.")

    embeddings = extract_features(train_paths, model, device, transform)
    mean = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False) + np.eye(embeddings.shape[1]) * 1e-5
    inv_cov = np.linalg.inv(cov)
    
    return model, device, mean, inv_cov, transform

def visualize(img_tensor, pred, dist):
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    fig, ax = plt.subplots()
    ax.imshow(img_np)
    ax.set_title(f"Prediction: {pred}\nDistance: {dist:.2f}")
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Streamlit App
st.title("Leather Surface Anomaly Detector")

train_folder = st.text_input("Enter path to training folder (with only 'good' images):", "data/leather/train/good")

if st.button("Load Model"):
    model, device, mean, inv_cov, transform = load_model_and_embeddings(train_folder)
    st.success("Model and embeddings loaded.")

uploaded_files = st.file_uploader("Upload test images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files and 'model' in locals():
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        img_tensor = transform(img).to(device)
        feat = model(img_tensor.unsqueeze(0)).cpu().numpy()[0]
        dist = mahalanobis_distance(feat, mean, inv_cov)

        # Hardcoded threshold or calculate dynamically
        threshold = 2.0
        pred = "Anomaly" if dist > threshold else "Good"

        # Visual
        buf = visualize(img_tensor.cpu(), pred, dist)
        st.image(buf, caption=f"{file.name} - {pred}", use_column_width=True)
else:
    st.info("Upload test images and load model to begin detection.")
