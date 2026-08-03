"""
Image Transformations

Defines the preprocessing and data augmentation pipelines used
during training, validation, testing, and inference.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from torchvision import transforms

from app.core.config import IMAGE_SIZE

# ==========================================================
# ImageNet Statistics
# (Required for EfficientNet pretrained weights)
# ==========================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================================
# Training Transform
# ==========================================================

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomVerticalFlip(p=0.2),

        transforms.RandomRotation(degrees=20),

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.85, 1.0),
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)

# ==========================================================
# Validation Transform
# ==========================================================

val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)

# ==========================================================
# Test Transform
# ==========================================================

test_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)

# ==========================================================
# Inference Transform
# ==========================================================

predict_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)

# ==========================================================
# Dictionary for Easy Access
# ==========================================================

TRANSFORMS = {
    "train": train_transform,
    "val": val_transform,
    "test": test_transform,
    "predict": predict_transform,
}