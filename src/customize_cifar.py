import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
import os
import pickle

# Configuration
SAVE_DIR = '../custom_cifar5'
os.makedirs(SAVE_DIR, exist_ok=True)

# Check if dataset already exists
if os.path.exists(os.path.join(SAVE_DIR, 'train_indices.pkl')):
    print("Loading saved dataset indices...")
    with open(os.path.join(SAVE_DIR, 'train_indices.pkl'), 'rb') as f:
        train_final_indices = pickle.load(f)
    with open(os.path.join(SAVE_DIR, 'val_indices.pkl'), 'rb') as f:
        val_final_indices = pickle.load(f)
    with open(os.path.join(SAVE_DIR, 'test_indices.pkl'), 'rb') as f:
        test_final_indices = pickle.load(f)
else:
    print("Creating new dataset...")
    # Load CIFAR10
    transform = transforms.ToTensor()
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Select only first 5 classes
    selected_classes = [0, 1, 2, 3, 4]
    
    # Filter training data
    train_indices = [i for i, (_, label) in enumerate(train_data) if label in selected_classes]
    
    # Filter test data
    test_indices = [i for i, (_, label) in enumerate(test_data) if label in selected_classes]
    
    # Create balanced splits: 1000 train + 200 val per class from training set
    train_final_indices = []
    val_final_indices = []
    test_final_indices = []
    
    np.random.seed(42)  # For reproducibility
    for class_label in selected_classes:
        # Get all indices for this class in training set
        class_train_indices = [i for i, idx in enumerate(train_indices) if train_data[train_indices[i]][1] == class_label]
        
        # Shuffle and split
        np.random.shuffle(class_train_indices)
        train_final_indices.extend(class_train_indices[:1000])
        val_final_indices.extend(class_train_indices[1000:1200])
        
        # Get all indices for this class in test set
        class_test_indices = [i for i, idx in enumerate(test_indices) if test_data[test_indices[i]][1] == class_label]
        np.random.shuffle(class_test_indices)
        test_final_indices.extend(class_test_indices[:500])
    
    # Save indices
    with open(os.path.join(SAVE_DIR, 'train_indices.pkl'), 'wb') as f:
        pickle.dump(train_final_indices, f)
    with open(os.path.join(SAVE_DIR, 'val_indices.pkl'), 'wb') as f:
        pickle.dump(val_final_indices, f)
    with open(os.path.join(SAVE_DIR, 'test_indices.pkl'), 'wb') as f:
        pickle.dump(test_final_indices, f)
    print(f"Dataset saved to {SAVE_DIR}")

# Load CIFAR10 for creating subsets
transform = transforms.ToTensor()
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create final subsets
train_final = Subset(train_data, train_final_indices)
val_final = Subset(train_data, val_final_indices)
test_final = Subset(test_data, test_final_indices)

print(f"Training set: {len(train_final)} images")
print(f"Validation set: {len(val_final)} images")
print(f"Test set: {len(test_final)} images")

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_final, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_final, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_final, batch_size=32, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")