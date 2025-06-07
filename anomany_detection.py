import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Dataset class to load images from folder
class LeatherDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(root_dir, fname))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path

# Feature extractor using pretrained ResNet18 (remove FC layer)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(model.children())[:-1])  # Remove last FC
        self.features.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, 1)
        return x

# Compute embeddings of all train images
def compute_train_embeddings(dataloader, model, device):
    embeddings = []
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        feats = model(imgs)
        embeddings.append(feats.cpu().numpy())
    embeddings = np.vstack(embeddings)
    return embeddings

# Calculate Mahalanobis distance between embedding and train distribution
def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
    return dist

# Save model parameters
def save_mahalanobis_model(mean, inv_cov, threshold, path="mahalanobis_model.pth"):
    torch.save({
        'mean': mean,
        'inv_cov': inv_cov,
        'threshold': threshold
    }, path)
    print(f"Model saved to {path}")

# Load model parameters
def load_mahalanobis_model(path="mahalanobis_model.pth"):
    data = torch.load(path)
    return data['mean'], data['inv_cov'], data['threshold']

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    train_dir = "data/leather/train/good"
    test_dir = "data/leather/test"

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # Load training dataset
    train_dataset = LeatherDataset(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    print(f"Loaded {len(train_dataset)} training images.")

    # Load test datasets
    test_folders = ['good', 'color', 'cut', 'fold']
    test_datasets = {}
    for folder in test_folders:
        path = os.path.join(test_dir, folder)
        test_datasets[folder] = LeatherDataset(path, transform=transform)

    # Initialize feature extractor
    feat_extractor = FeatureExtractor().to(device)

    # Compute train embeddings
    train_embeddings = compute_train_embeddings(train_loader, feat_extractor, device)
    print(f"Computed train embeddings shape: {train_embeddings.shape}")

    # Compute mean, covariance and inverse covariance
    mean_emb = np.mean(train_embeddings, axis=0)
    cov_emb = np.cov(train_embeddings, rowvar=False)
    cov_emb += np.eye(cov_emb.shape[0]) * 1e-5  # regularization
    inv_cov_emb = np.linalg.inv(cov_emb)

    # Temporary save without threshold
    save_mahalanobis_model(mean_emb, inv_cov_emb, threshold=None)

    # Compute test distances
    all_test_distances = []
    for label, dataset in test_datasets.items():
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        distances = []
        for imgs, paths in loader:
            imgs = imgs.to(device)
            feats = feat_extractor(imgs).cpu().numpy()
            for feat in feats:
                dist = mahalanobis_distance(feat, mean_emb, inv_cov_emb)
                distances.append(dist)
                all_test_distances.append((dist, label))
        print(f"{label} - {len(distances)} images")

    # Determine threshold from 'good' test images
    good_distances = [dist for dist, lbl in all_test_distances if lbl == 'good']
    threshold = np.percentile(good_distances, 95)
    print(f"Set anomaly threshold at 95th percentile distance: {threshold:.4f}")

    # Update model file with threshold
    save_mahalanobis_model(mean_emb, inv_cov_emb, threshold)

    # Visualization
    def visualize_samples(dataset, label, num=5):
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        plt.figure(figsize=(15, 5))
        count = 0
        for img, path in loader:
            img_tensor = img.to(device)
            feat = feat_extractor(img_tensor).cpu().numpy()[0]
            dist = mahalanobis_distance(feat, mean_emb, inv_cov_emb)
            pred = "Anomaly" if dist > threshold else "Good"
            img_show = img[0].permute(1, 2, 0).numpy()
            img_show = img_show * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_show = np.clip(img_show, 0, 1)

            plt.subplot(1, num, count + 1)
            plt.imshow(img_show)
            plt.title(f"True: {label}\nPred: {pred}\nDist: {dist:.2f}")
            plt.axis('off')
            count += 1
            if count == num:
                break
        plt.show()

    # Show examples from each class
    for label, dataset in test_datasets.items():
        print(f"Visualizing results for {label} samples:")
        visualize_samples(dataset, label, num=5)

if __name__ == "__main__":
    main()
