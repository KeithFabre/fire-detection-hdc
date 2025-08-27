import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# VGG16 expects 224x224 images
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

def get_dataloaders(data_dir, batch_size=BATCH_SIZE, val_split=0.2, num_workers=4):
    """
    Creates training and validation dataloaders optimized for VGG16.
    
    Args:
        data_dir (str): Path to the root data directory (e.g., '../Training').
        batch_size (int): The number of samples per batch.
        val_split (float): The proportion of the dataset to use for validation.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, class_names).
    """
    # VGG16 preprocessing: resize to 224x224 and normalize with ImageNet stats
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the entire dataset using ImageFolder
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Class to index mapping: {full_dataset.class_to_idx}")

    # Split dataset into training and validation sets
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Ensure consistent splitting
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"Total images: {total_size}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # Create DataLoaders for training and validation
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, class_names

if __name__ == '__main__':
    # Example of how to use this script
    data_directory = '../Training'
    train_loader, val_loader, class_names = get_dataloaders(data_directory)

    # Fetch and print details from one batch to verify
    images, labels = next(iter(train_loader))
    print("\n--- Verification ---")
    print(f"Batch of images has shape: {images.shape}")
    print(f"Batch of labels has shape: {labels.shape}")
    print(f"A label from the batch: {labels[0].item()}")
    print(f"Corresponding class: {class_names[labels[0].item()]}")
    print("--------------------") 