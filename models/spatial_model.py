"""
Spatial Feature Extractor using CNN
Extracts frame-level visual features for deepfake detection

Based on EfficientNet and XceptionNet architectures
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import timm


class SpatialFeatureExtractor(nn.Module):
    """
    CNN-based spatial feature extractor for frame-level analysis
    Supports multiple backbone architectures
    """
    
    def __init__(self, backbone='efficientnet_b4', pretrained=True, num_classes=2, freeze_backbone=False):
        """
        Initialize spatial feature extractor
        
        Args:
            backbone: CNN architecture ('efficientnet_b4', 'xception', 'resnet50')
            pretrained: Use ImageNet pretrained weights
            num_classes: Number of output classes (2 for binary classification)
            freeze_backbone: Freeze backbone layers for transfer learning
        """
        super(SpatialFeatureExtractor, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Load backbone
        if backbone == 'efficientnet_b4':
            self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
            self.feature_dim = 1792
        elif backbone == 'efficientnet_b0':
            self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
            self.feature_dim = 1280
        elif backbone == 'xception':
            self.backbone = timm.create_model('xception', pretrained=pretrained, num_classes=0)
            self.feature_dim = 2048
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Activation for features
        self.feature_extractor = nn.Sequential(
            self.backbone,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
        
        Returns:
            logits: Classification logits (batch_size, num_classes)
            features: Extracted features (batch_size, feature_dim)
        """
        # Extract features
        if self.backbone_name == 'resnet50':
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
        else:
            features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits, features
    
    def extract_features(self, x):
        """
        Extract only features without classification
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
        
        Returns:
            features: Extracted features (batch_size, feature_dim)
        """
        with torch.no_grad():
            if self.backbone_name == 'resnet50':
                features = self.backbone(x)
                features = features.view(features.size(0), -1)
            else:
                features = self.backbone(x)
        return features


def get_spatial_transforms(mode='train', image_size=224):
    """
    Get image preprocessing transforms
    
    Args:
        mode: 'train' or 'val'
        image_size: Target image size
    
    Returns:
        transforms: Composed transforms
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


if __name__ == '__main__':
    # Test spatial model
    print("Testing Spatial Feature Extractor...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = SpatialFeatureExtractor(backbone='efficientnet_b0', pretrained=False)
    model = model.to(device)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    logits, features = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nSpatial Feature Extractor test passed!")
