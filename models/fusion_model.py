"""
Multimodal Fusion Model
Combines spatial and temporal features for robust deepfake detection

Implements late fusion architecture
"""

import torch
import torch.nn as nn
import numpy as np
from .spatial_model import SpatialFeatureExtractor
from .temporal_model import TemporalAnalyzer


class MultimodalDetector(nn.Module):
    """
    Multimodal deepfake detector combining spatial and temporal analysis
    """
    
    def __init__(self, spatial_backbone='efficientnet_b0', temporal_type='lstm',
                 spatial_feature_dim=1280, temporal_hidden_dim=512,
                 num_classes=2, fusion_type='concat', dropout=0.3):
        """
        Initialize multimodal detector
        
        Args:
            spatial_backbone: Backbone for spatial model
            temporal_type: Type of temporal model ('lstm', 'transformer')
            spatial_feature_dim: Feature dimension from spatial model
            temporal_hidden_dim: Hidden dimension for temporal model
            num_classes: Number of output classes
            fusion_type: Feature fusion strategy ('concat', 'add', 'attention')
            dropout: Dropout rate
        """
        super(MultimodalDetector, self).__init__()
        
        self.fusion_type = fusion_type
        
        # Spatial feature extractor
        self.spatial_model = SpatialFeatureExtractor(
            backbone=spatial_backbone,
            pretrained=True,
            num_classes=num_classes
        )
        
        # Temporal analyzer
        if temporal_type == 'lstm':
            self.temporal_model = TemporalAnalyzer(
                input_dim=spatial_feature_dim,
                hidden_dim=temporal_hidden_dim,
                num_classes=num_classes,
                model_type='lstm'
            )
            temporal_feature_dim = temporal_hidden_dim * 2  # Bidirectional
        else:
            from .temporal_model import TemporalTransformer
            self.temporal_model = TemporalTransformer(
                input_dim=spatial_feature_dim,
                hidden_dim=temporal_hidden_dim,
                num_classes=num_classes
            )
            temporal_feature_dim = temporal_hidden_dim
        
        # Fusion layer
        if fusion_type == 'concat':
            fusion_dim = spatial_feature_dim + temporal_feature_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
        elif fusion_type == 'add':
            # Project both to same dimension
            self.spatial_proj = nn.Linear(spatial_feature_dim, 512)
            self.temporal_proj = nn.Linear(temporal_feature_dim, 512)
            self.fusion_layer = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.spatial_attention = nn.Linear(spatial_feature_dim, 1)
            self.temporal_attention = nn.Linear(temporal_feature_dim, 1)
            fusion_dim = spatial_feature_dim + temporal_feature_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, frames):
        """
        Forward pass through multimodal detector
        
        Args:
            frames: Video frames (batch_size, num_frames, C, H, W)
        
        Returns:
            logits: Classification logits
            spatial_features: Spatial features
            temporal_features: Temporal features
        """
        batch_size, num_frames, C, H, W = frames.shape
        
        # Reshape for spatial processing
        frames_flat = frames.view(-1, C, H, W)  # (batch*frames, C, H, W)
        
        # Extract spatial features for each frame
        with torch.no_grad():
            spatial_logits, spatial_features = self.spatial_model(frames_flat)
        
        # Reshape back to sequences
        spatial_features = spatial_features.view(batch_size, num_frames, -1)
        
        # Extract temporal features
        temporal_logits, temporal_features = self.temporal_model(spatial_features)
        
        # Aggregate spatial features (mean pooling)
        spatial_features_agg = torch.mean(spatial_features, dim=1)
        
        # Fusion
        if self.fusion_type == 'concat':
            fused_features = torch.cat([spatial_features_agg, temporal_features], dim=1)
            logits = self.fusion_layer(fused_features)
        elif self.fusion_type == 'add':
            spatial_proj = self.spatial_proj(spatial_features_agg)
            temporal_proj = self.temporal_proj(temporal_features)
            fused_features = spatial_proj + temporal_proj
            logits = self.fusion_layer(fused_features)
        elif self.fusion_type == 'attention':
            spatial_weight = torch.sigmoid(self.spatial_attention(spatial_features_agg))
            temporal_weight = torch.sigmoid(self.temporal_attention(temporal_features))
            
            spatial_weighted = spatial_features_agg * spatial_weight
            temporal_weighted = temporal_features * temporal_weight
            
            fused_features = torch.cat([spatial_weighted, temporal_weighted], dim=1)
            logits = self.fusion_layer(fused_features)
        
        return logits, spatial_features_agg, temporal_features
    
    def predict(self, frames):
        """
        Make prediction on video frames
        
        Args:
            frames: Numpy array or torch tensor of frames (num_frames, H, W, C)
        
        Returns:
            prediction: Class prediction (0=real, 1=fake)
            confidence: Confidence score
            spatial_features: Spatial features
            temporal_features: Temporal features
        """
        self.eval()
        
        # Convert to tensor if needed
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames).float()
        
        # Add batch dimension if needed
        if frames.dim() == 4:  # (num_frames, C, H, W)
            frames = frames.unsqueeze(0)  # (1, num_frames, C, H, W)
        
        # Ensure correct dimension order (batch, frames, C, H, W)
        if frames.shape[-1] == 3:  # (batch, frames, H, W, C)
            frames = frames.permute(0, 1, 4, 2, 3)
        
        # Move to appropriate device
        device = next(self.parameters()).device
        frames = frames.to(device)
        
        with torch.no_grad():
            logits, spatial_features, temporal_features = self.forward(frames)
            
            # Get prediction
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            return (
                prediction.item(),
                confidence.item(),
                spatial_features.cpu().numpy(),
                temporal_features.cpu().numpy()
            )


class SimpleMultimodalDetector:
    """
    Simplified interface for multimodal detection
    Handles model loading and inference
    """
    
    def __init__(self, config, device='cpu'):
        """
        Initialize detector
        
        Args:
            config: Configuration object
            device: Torch device
        """
        self.config = config
        self.device = device
        
        # Initialize model
        self.model = MultimodalDetector(
            spatial_backbone='efficientnet_b0',
            temporal_type='lstm',
            fusion_type='concat'
        ).to(device)
        
        # Load pretrained weights if available
        self.load_weights()
    
    def load_weights(self):
        """Load pretrained model weights"""
        try:
            if hasattr(self.config, 'FUSION_MODEL_PATH'):
                import os
                if os.path.exists(self.config.FUSION_MODEL_PATH):
                    checkpoint = torch.load(
                        self.config.FUSION_MODEL_PATH,
                        map_location=self.device
                    )
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model weights from {self.config.FUSION_MODEL_PATH}")
                else:
                    print("No pretrained weights found. Using randomly initialized model.")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
    
    def predict(self, frames):
        """
        Predict deepfake on video frames
        
        Args:
            frames: Preprocessed video frames
        
        Returns:
            prediction, confidence, spatial_features, temporal_features
        """
        return self.model.predict(frames)


if __name__ == '__main__':
    # Test multimodal detector
    print("Testing Multimodal Detector...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = MultimodalDetector(
        spatial_backbone='efficientnet_b0',
        temporal_type='lstm',
        fusion_type='concat'
    )
    model = model.to(device)
    
    # Test forward pass
    batch_size = 2
    num_frames = 30
    C, H, W = 3, 224, 224
    
    dummy_input = torch.randn(batch_size, num_frames, C, H, W).to(device)
    
    logits, spatial_feat, temporal_feat = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Spatial features shape: {spatial_feat.shape}")
    print(f"Temporal features shape: {temporal_feat.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nMultimodal Detector test passed!")
