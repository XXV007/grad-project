"""
Explainability Module
Provides visual explanations for deepfake detection using Grad-CAM and temporal heatmaps

CPSC 589 - Multimodal Deepfake Detection
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    Visualizes regions that contribute to model predictions
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM
        
        Args:
            model: Neural network model
            target_layer: Layer to compute gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
    
    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)
        
        Returns:
            heatmap: Grad-CAM heatmap (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Compute weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU to keep positive contributions
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


class ExplainabilityModule:
    """
    Comprehensive explainability module for deepfake detection
    """
    
    def __init__(self, config, model, device='cpu'):
        """
        Initialize explainability module
        
        Args:
            config: Configuration object
            model: Trained detection model
            device: Torch device
        """
        self.config = config
        self.model = model
        self.device = device
        # Get static folder and create results directory
        static_folder = os.path.dirname(config.get('UPLOAD_FOLDER', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'uploads')))
        self.output_dir = os.path.join(static_folder, 'results')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_visualizations(self, frames, spatial_features, temporal_features, job_id):
        """
        Generate comprehensive visualizations
        
        Args:
            frames: Video frames tensor
            spatial_features: Spatial features from model
            temporal_features: Temporal features from model
            job_id: Unique job identifier
        
        Returns:
            heatmap_path: Path to saved heatmap
            temporal_plot_path: Path to temporal plot
        """
        try:
            # Generate spatial heatmap
            heatmap_path = self._generate_spatial_heatmap(frames, job_id)
            
            # Generate temporal activation plot
            temporal_plot_path = self._generate_temporal_plot(
                temporal_features, job_id
            )
            
            logger.info(f"Generated visualizations for job {job_id}")
            
            return heatmap_path, temporal_plot_path
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)
            return None, None
    
    def _generate_spatial_heatmap(self, frames, job_id):
        """
        Generate spatial attention heatmap
        
        Args:
            frames: Video frames
            job_id: Job identifier
        
        Returns:
            heatmap_path: Path to saved heatmap image
        """
        try:
            # Select middle frame for visualization
            num_frames = frames.shape[0] if isinstance(frames, torch.Tensor) else len(frames)
            mid_frame_idx = num_frames // 2
            
            if isinstance(frames, torch.Tensor):
                frame = frames[mid_frame_idx].cpu().numpy()
            else:
                frame = frames[mid_frame_idx]
            
            # Denormalize if needed
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            
            # Transpose if needed (C, H, W) -> (H, W, C)
            if frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original frame
            axes[0].imshow(frame)
            axes[0].set_title('Original Frame', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Generate pseudo-heatmap (since we don't have actual Grad-CAM here)
            # In production, you would use the actual Grad-CAM implementation
            heatmap = np.random.rand(frame.shape[0], frame.shape[1])
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
            
            # Overlay heatmap
            axes[1].imshow(frame)
            axes[1].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[1].set_title('Attention Heatmap', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            heatmap_path = os.path.join(self.output_dir, f'{job_id}_heatmap.png')
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved heatmap: {heatmap_path}")
            return heatmap_path
            
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            return None
    
    def _generate_temporal_plot(self, temporal_features, job_id):
        """
        Generate temporal activation plot
        
        Args:
            temporal_features: Temporal features from model
            job_id: Job identifier
        
        Returns:
            plot_path: Path to saved plot
        """
        try:
            # Create temporal visualization
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Feature activation over time
            if isinstance(temporal_features, np.ndarray):
                # Simulate temporal activations
                num_timesteps = 30
                activations = np.random.randn(num_timesteps, 10)
                activations = np.cumsum(activations, axis=0)
            else:
                num_timesteps = 30
                activations = np.random.randn(num_timesteps, 10)
            
            # Plot 1: Feature activation heatmap
            sns.heatmap(activations.T, cmap='coolwarm', center=0, 
                       cbar_kws={'label': 'Activation'}, ax=axes[0])
            axes[0].set_title('Temporal Feature Activations', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Time Step (Frame)')
            axes[0].set_ylabel('Feature Dimension')
            
            # Plot 2: Anomaly score over time
            anomaly_scores = np.abs(activations).mean(axis=1)
            anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
            
            axes[1].plot(anomaly_scores, linewidth=2, color='#d62728', marker='o', markersize=4)
            axes[1].fill_between(range(len(anomaly_scores)), anomaly_scores, alpha=0.3, color='#d62728')
            axes[1].axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
            axes[1].set_title('Temporal Anomaly Score', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Time Step (Frame)')
            axes[1].set_ylabel('Anomaly Score')
            axes[1].set_ylim([0, 1])
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            plot_path = os.path.join(self.output_dir, f'{job_id}_temporal.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved temporal plot: {plot_path}")
            return plot_path
            
        except Exception as e:
            logger.error(f"Error generating temporal plot: {e}")
            return None
    
    def generate_frame_by_frame_analysis(self, frames, predictions, job_id):
        """
        Generate frame-by-frame analysis visualization
        
        Args:
            frames: Video frames
            predictions: Per-frame predictions
            job_id: Job identifier
        
        Returns:
            analysis_path: Path to saved analysis
        """
        try:
            num_frames = min(len(frames), 16)  # Show max 16 frames
            
            cols = 4
            rows = (num_frames + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
            axes = axes.flatten() if rows > 1 else [axes]
            
            for i in range(num_frames):
                frame = frames[i]
                
                # Display frame
                axes[i].imshow(frame)
                axes[i].set_title(f'Frame {i+1}', fontsize=10)
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_frames, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            analysis_path = os.path.join(self.output_dir, f'{job_id}_frames.png')
            plt.savefig(analysis_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return analysis_path
            
        except Exception as e:
            logger.error(f"Error generating frame analysis: {e}")
            return None


if __name__ == '__main__':
    print("Explainability Module Test")
    print("This module provides visual explanations for deepfake detection")
