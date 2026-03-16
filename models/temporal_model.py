"""
Temporal Analyzer using LSTM and 3D-CNN
Captures motion dynamics and temporal inconsistencies

Analyzes sequential frames to detect deepfake artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAnalyzer(nn.Module):
    """
    Temporal feature analyzer for motion-based deepfake detection
    Uses LSTM or 3D-CNN to model temporal patterns
    """
    
    def __init__(self, input_dim=1280, hidden_dim=512, num_layers=2, 
                 num_classes=2, dropout=0.3, model_type='lstm'):
        """
        Initialize temporal analyzer
        
        Args:
            input_dim: Dimension of spatial features
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            model_type: 'lstm' or '3dcnn'
        """
        super(TemporalAnalyzer, self).__init__()
        
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if model_type == 'lstm':
            # LSTM-based temporal model
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            
            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
            
        elif model_type == '3dcnn':
            # 3D-CNN based temporal model
            # Input: (batch, channels, depth, height, width)
            self.conv3d_1 = nn.Conv3d(input_dim, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.bn1 = nn.BatchNorm3d(256)
            self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
            
            self.conv3d_2 = nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.bn2 = nn.BatchNorm3d(128)
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
            
            self.conv3d_3 = nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.bn3 = nn.BatchNorm3d(64)
            
            self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
               For LSTM: (batch_size, sequence_length, input_dim)
               For 3D-CNN: (batch_size, channels, depth, height, width)
        
        Returns:
            logits: Classification logits (batch_size, num_classes)
            temporal_features: Extracted temporal features
        """
        if self.model_type == 'lstm':
            return self._forward_lstm(x)
        elif self.model_type == '3dcnn':
            return self._forward_3dcnn(x)
    
    def _forward_lstm(self, x):
        """
        LSTM forward pass with attention
        
        Args:
            x: (batch_size, sequence_length, input_dim)
        
        Returns:
            logits, temporal_features
        """
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out)  # (batch, seq, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        temporal_features = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden*2)
        
        # Classification
        logits = self.classifier(temporal_features)
        
        return logits, temporal_features
    
    def _forward_3dcnn(self, x):
        """
        3D-CNN forward pass
        
        Args:
            x: (batch_size, channels, depth, height, width)
        
        Returns:
            logits, temporal_features
        """
        # 3D convolutions
        x = F.relu(self.bn1(self.conv3d_1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv3d_2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3d_3(x)))
        
        # Global pooling
        x = self.global_pool(x)
        temporal_features = x.view(x.size(0), -1)
        
        # Classification
        logits = self.classifier(temporal_features)
        
        return logits, temporal_features


class TemporalTransformer(nn.Module):
    """
    Transformer-based temporal analyzer (alternative to LSTM)
    """
    
    def __init__(self, input_dim=1280, hidden_dim=512, num_heads=8, 
                 num_layers=4, num_classes=2, dropout=0.3):
        """
        Initialize temporal transformer
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(TemporalTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, sequence_length, input_dim)
        
        Returns:
            logits, temporal_features
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Aggregate temporal features (mean pooling)
        temporal_features = torch.mean(x, dim=1)
        
        # Classification
        logits = self.classifier(temporal_features)
        
        return logits, temporal_features


if __name__ == '__main__':
    # Test temporal models
    print("Testing Temporal Analyzer...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test LSTM model
    print("\n1. Testing LSTM Model:")
    lstm_model = TemporalAnalyzer(input_dim=1280, hidden_dim=512, model_type='lstm')
    lstm_model = lstm_model.to(device)
    
    batch_size, seq_len, input_dim = 4, 30, 1280
    dummy_input = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    logits, features = lstm_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    # Test Transformer model
    print("\n2. Testing Transformer Model:")
    transformer_model = TemporalTransformer(input_dim=1280, hidden_dim=512)
    transformer_model = transformer_model.to(device)
    
    logits, features = transformer_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    
    print("\nTemporal Analyzer test passed!")
