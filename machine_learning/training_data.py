import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Load training data from CSV file
def load_training_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess text data using TF-IDF vectorization
def preprocess_text_data(data, column_name):
    vectorizer = TfidfVectorizer(max_features=5000)
    text_data = data[column_name]
    vectorized_data = vectorizer.fit_transform(text_data)
    return vectorized_data

# Preprocess image data using patch extraction and resizing
def preprocess_image_data(data, column_name, image_size):
    image_data = []
    for file_name in data[column_name]:
        image_path = os.path.join('images', file_name)
        image = cv2.imread(image_path)
        patches = extract_patches_2d(image, patch_size=(image_size, image_size), max_patches=10)
        for patch in patches:
            image_data.append(cv2.resize(patch, (image_size, image_size)))
    image_data = np.array(image_data)
    return image_data

# Split data into training and validation sets
def split_data(data, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(data.drop(['target'], axis=1), data['target'], test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val

# Create a custom dataset class for loading and preprocessing data
class TrainingDataset(Dataset):
    def __init__(self, data, column_name, image_size, transform=None):
        self.data = data
        self.column_name = column_name
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx, 0]
        image_path = os.path.join('images', file_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))
        if self.transform:
            image = self.transform(image)
        label = self.data.iloc[idx, 1]
        return image, label

# Create a data loader for the custom dataset
def create_data_loader(data, column_name, image_size, batch_size, transform=None):
    dataset = TrainingDataset(data, column_name, image_size, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# Example usage
if __name__ == "__main__":
    # Load training data from CSV file
    data = load_training_data('training_data.csv')
    
    # Preprocess text data using TF-IDF vectorization
    text_data = preprocess_text_data(data, 'text_column')
    
    # Preprocess image data using patch extraction and resizing
    image_data = preprocess_image_data(data, 'image_column', image_size=224)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(data)
    
    # Create a custom dataset class for loading and preprocessing data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_loader = create_data_loader(data, 'image_column', image_size=224, batch_size=32, transform=transform)
    
    # Iterate over the data loader
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"Batch {batch_idx}: {images.shape}, {labels.shape}")
