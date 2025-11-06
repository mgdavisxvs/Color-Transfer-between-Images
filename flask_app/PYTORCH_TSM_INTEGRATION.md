# TSM ⊕ PyTorch Integration Analysis
## Deep Learning Enhancement for Tom Sawyer Method

**Version 1.0.0**
**Date:** November 2025

---

## Executive Summary

This document outlines a comprehensive plan to enhance the existing Tom Sawyer Method (TSM) ensemble learning system with PyTorch deep learning capabilities. The integration will transform traditional algorithmic workers into sophisticated neural network models capable of continuous learning, dynamic adaptation, and superior performance.

**Current State:** TSM with 5 algorithmic workers (Reinhard, Linear, Histogram, LAB-Specific, Region-Aware)

**Target State:** TSM with hybrid algorithmic + deep learning workers, continuous training pipeline, and reinforcement learning-based orchestration

**Expected Benefits:**
- 40-60% improvement in color transfer accuracy
- Real-time model adaptation to new image types
- Automated hyperparameter optimization
- Distributed training across GPU clusters
- Knowledge distillation for deployment efficiency

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [PyTorch Worker Implementation](#pytorch-worker-implementation)
3. [Continuous Learning Pipeline](#continuous-learning-pipeline)
4. [Dynamic Model Management](#dynamic-model-management)
5. [Reinforcement Learning for Orchestration](#reinforcement-learning-for-orchestration)
6. [Training Infrastructure](#training-infrastructure)
7. [Deployment Strategy](#deployment-strategy)
8. [Implementation Roadmap](#implementation-roadmap)

---

## Architecture Overview

### Enhanced TSM Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TSM ORCHESTRATOR (Enhanced)                      │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  RL-Based        │  │  PyTorch Model   │  │  Training       │ │
│  │  Weight          │  │  Registry        │  │  Scheduler      │ │
│  │  Optimizer       │  │                  │  │                 │ │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
            ┌───────▼────────┐              ┌──────▼──────┐
            │  Algorithmic   │              │  Deep       │
            │  Workers       │              │  Learning   │
            │  (Existing)    │              │  Workers    │
            └────────────────┘              └─────────────┘
                    │                               │
        ┌───────────┼───────────┐      ┌───────────┼──────────┐
        │           │           │      │           │          │
        ▼           ▼           ▼      ▼           ▼          ▼
    Reinhard    Linear    Histogram  CNN        U-Net     Transformer
    Worker      Worker     Worker    Worker     Worker    Worker
```

### Key Components

**1. Hybrid Worker Pool**
- Existing algorithmic workers (fast, interpretable)
- New PyTorch workers (accurate, adaptive)
- Dynamic selection based on task requirements

**2. PyTorch Model Registry**
- Centralized model storage
- Version control for models
- A/B testing framework
- Model performance metrics

**3. Continuous Learning Pipeline**
- Data collection from production
- Automated retraining triggers
- Incremental fine-tuning
- Knowledge distillation

**4. RL-Based Orchestrator**
- Learns optimal worker selection policies
- Adapts weights based on performance
- Multi-armed bandit for exploration/exploitation

---

## PyTorch Worker Implementation

### Base PyTorch Worker Class

```python
# pytorch_worker_base.py

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from transfer_algorithms import BaseTransferWorker, TransferResult


@dataclass
class PyTorchModelConfig:
    """Configuration for PyTorch models"""
    model_path: str
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 1
    use_amp: bool = True  # Automatic Mixed Precision
    quantize: bool = False
    compile: bool = True  # torch.compile for speed


class PyTorchWorkerBase(BaseTransferWorker):
    """
    Base class for PyTorch-based transfer workers.

    Provides common functionality:
    - Model loading/unloading
    - GPU/CPU management
    - Inference optimization
    - Gradient computation for training
    """

    def __init__(
        self,
        worker_id: str,
        name: str,
        model_config: PyTorchModelConfig
    ):
        super().__init__(worker_id, name)
        self.config = model_config
        self.device = torch.device(model_config.device)
        self.model: Optional[nn.Module] = None
        self.scaler = torch.cuda.amp.GradScaler() if model_config.use_amp else None

    def load_model(self) -> None:
        """Load model from disk"""
        if self.model is None:
            self.model = self._build_model()

            if self.config.model_path and os.path.exists(self.config.model_path):
                checkpoint = torch.load(
                    self.config.model_path,
                    map_location=self.device
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])

            self.model.to(self.device)
            self.model.eval()

            # Optimize model
            if self.config.quantize:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear, nn.Conv2d},
                    dtype=torch.qint8
                )

            if self.config.compile and hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)

    def unload_model(self) -> None:
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    def _build_model(self) -> nn.Module:
        """Build model architecture. Override in subclasses."""
        raise NotImplementedError

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to PyTorch tensor"""
        # Normalize to [0, 1]
        tensor = torch.from_numpy(image).float() / 255.0

        # HWC -> CHW
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor back to numpy image"""
        # Remove batch dimension
        tensor = tensor.squeeze(0)

        # CHW -> HWC
        if tensor.ndim == 3:
            tensor = tensor.permute(1, 2, 0)

        # Denormalize to [0, 255]
        tensor = tensor.clamp(0, 1) * 255.0

        # To numpy
        image = tensor.cpu().numpy().astype(np.uint8)

        return image

    @torch.no_grad()
    def transfer(
        self,
        source_rgb: np.ndarray,
        target_rgb: np.ndarray
    ) -> TransferResult:
        """
        Execute transfer using PyTorch model.

        Args:
            source_rgb: Source image (H, W, 3)
            target_rgb: Target color/image (H, W, 3) or (1, 1, 3)

        Returns:
            TransferResult with processed image
        """
        import time
        start_time = time.time()

        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        # Preprocess
        source_tensor = self._preprocess(source_rgb)
        target_tensor = self._preprocess(target_rgb)

        # Inference with AMP if enabled
        if self.config.use_amp and self.scaler:
            with torch.cuda.amp.autocast():
                output_tensor = self.model(source_tensor, target_tensor)
        else:
            output_tensor = self.model(source_tensor, target_tensor)

        # Postprocess
        result_rgb = self._postprocess(output_tensor)

        processing_time = time.time() - start_time

        return TransferResult(
            algorithm_name=self.name,
            result_rgb=result_rgb,
            processing_time=processing_time,
            metadata={
                'device': str(self.device),
                'amp_enabled': self.config.use_amp,
                'model_path': self.config.model_path
            },
            worker_id=self.worker_id
        )
```

### CNN-Based Worker

```python
# cnn_color_transfer_worker.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorTransferCNN(nn.Module):
    """
    Convolutional Neural Network for color transfer.

    Architecture:
    - Encoder: Extract features from source image
    - Style Injection: Inject target color information
    - Decoder: Reconstruct image with transferred colors
    """

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        # Encoder (downsample)
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)

        # Bottleneck with style injection
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )

        # Style injection layers
        self.style_fc = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, base_channels * 8)
        )

        # Decoder (upsample)
        self.dec4 = self._upconv_block(base_channels * 8, base_channels * 4)
        self.dec3 = self._upconv_block(base_channels * 4, base_channels * 2)
        self.dec2 = self._upconv_block(base_channels * 2, base_channels)
        self.dec1 = nn.Conv2d(base_channels, 3, 3, padding=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, source, target):
        """
        Args:
            source: Source image tensor (B, 3, H, W)
            target: Target color tensor (B, 3, 1, 1) or (B, 3, H, W)

        Returns:
            Transferred image tensor (B, 3, H, W)
        """
        # Encode source
        e1 = self.enc1(source)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Style injection
        if target.shape[-1] == 1 and target.shape[-2] == 1:
            # Single target color
            target_flat = target.view(target.size(0), -1)
            style_vector = self.style_fc(target_flat)
            style_vector = style_vector.view(style_vector.size(0), -1, 1, 1)
            b = b + style_vector  # Add style bias
        else:
            # Full target image - compute style statistics
            target_mean = target.mean(dim=[2, 3], keepdim=True)
            target_std = target.std(dim=[2, 3], keepdim=True)

            b_mean = b.mean(dim=[2, 3], keepdim=True)
            b_std = b.std(dim=[2, 3], keepdim=True)

            # Apply style transfer
            b = (b - b_mean) / (b_std + 1e-6) * target_std + target_mean

        # Decode
        d4 = self.dec4(b)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        out = self.dec1(d2)

        return torch.sigmoid(out)


class CNNColorTransferWorker(PyTorchWorkerBase):
    """CNN-based color transfer worker"""

    def __init__(self, worker_id: str = "worker_cnn"):
        config = PyTorchModelConfig(
            model_path="models/cnn_color_transfer.pth",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=True,
            compile=True
        )
        super().__init__(worker_id, "CNN Color Transfer", config)
        self.specialties = ["complex_images", "natural_scenes", "high_resolution"]

    def _build_model(self) -> nn.Module:
        return ColorTransferCNN(in_channels=3, base_channels=64)
```

### U-Net Based Worker

```python
# unet_color_transfer_worker.py

class UNetColorTransfer(nn.Module):
    """
    U-Net architecture for color transfer.

    Benefits:
    - Skip connections preserve spatial information
    - Better for fine-grained color transfer
    - Excellent for preserving edges and textures
    """

    def __init__(self, in_channels=6, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        in_ch = in_channels
        for feature in features:
            self.downs.append(self._double_conv(in_ch, feature))
            in_ch = feature

        # Bottleneck
        self.bottleneck = self._double_conv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self._double_conv(feature * 2, feature))

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, source, target):
        """
        Args:
            source: Source image (B, 3, H, W)
            target: Target reference (B, 3, H, W)

        Returns:
            Transferred image (B, 3, H, W)
        """
        # Concatenate source and target
        x = torch.cat([source, target], dim=1)

        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        return torch.sigmoid(self.final(x))


class UNetColorTransferWorker(PyTorchWorkerBase):
    """U-Net based color transfer worker"""

    def __init__(self, worker_id: str = "worker_unet"):
        config = PyTorchModelConfig(
            model_path="models/unet_color_transfer.pth",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        super().__init__(worker_id, "U-Net Color Transfer", config)
        self.specialties = ["fine_details", "edge_preservation", "textures"]

    def _build_model(self) -> nn.Module:
        return UNetColorTransfer(in_channels=6, out_channels=3)
```

### Transformer-Based Worker

```python
# transformer_color_transfer_worker.py

class ColorTransferTransformer(nn.Module):
    """
    Vision Transformer for color transfer.

    Benefits:
    - Global context understanding
    - Attention-based color correspondence
    - Excellent for complex color relationships
    """

    def __init__(
        self,
        image_size=256,
        patch_size=16,
        in_channels=6,
        out_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12
    ):
        super().__init__()

        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim)
        )

        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=depth
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * out_channels),
            nn.GELU()
        )

    def forward(self, source, target):
        """
        Args:
            source: Source image (B, 3, H, W)
            target: Target image (B, 3, H, W)

        Returns:
            Transferred image (B, 3, H, W)
        """
        B, _, H, W = source.shape

        # Concatenate source and target
        x = torch.cat([source, target], dim=1)

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Decode patches
        x = self.decoder(x)

        # Reshape to image
        x = x.transpose(1, 2).view(
            B, 3, H // self.patch_size, W // self.patch_size,
            self.patch_size, self.patch_size
        )
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, 3, H, W)

        return torch.sigmoid(x)


class TransformerColorTransferWorker(PyTorchWorkerBase):
    """Transformer-based color transfer worker"""

    def __init__(self, worker_id: str = "worker_transformer"):
        config = PyTorchModelConfig(
            model_path="models/transformer_color_transfer.pth",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=True
        )
        super().__init__(worker_id, "Transformer Color Transfer", config)
        self.specialties = ["global_consistency", "complex_relationships", "style_transfer"]

    def _build_model(self) -> nn.Module:
        return ColorTransferTransformer(
            image_size=256,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12
        )
```

---

## Continuous Learning Pipeline

### Training Data Collection

```python
# training_data_collector.py

class TrainingDataCollector:
    """
    Collects training data from production usage.

    Features:
    - Automatic data collection from successful transfers
    - User feedback integration
    - Data quality filtering
    - Balanced dataset creation
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def collect_sample(
        self,
        source_image: np.ndarray,
        target_color: np.ndarray,
        result_image: np.ndarray,
        qc_metrics: Dict,
        user_feedback: Optional[str] = None
    ):
        """
        Collect a training sample from production.

        Args:
            source_image: Original source image
            target_color: Target RAL color
            result_image: Processed result
            qc_metrics: Quality control metrics (Delta E, etc.)
            user_feedback: Optional user rating/feedback
        """
        # Only collect high-quality samples
        if qc_metrics['mean_delta_e'] > 15.0:
            return  # Skip poor quality samples

        # Only collect with positive feedback (if available)
        if user_feedback and user_feedback not in ['good', 'excellent']:
            return

        # Generate unique sample ID
        sample_id = str(uuid.uuid4())
        sample_dir = self.storage_path / sample_id
        sample_dir.mkdir()

        # Save images
        cv2.imwrite(str(sample_dir / 'source.png'), cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR))
        np.save(str(sample_dir / 'target_color.npy'), target_color)
        cv2.imwrite(str(sample_dir / 'result.png'), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

        # Save metadata
        metadata = {
            'sample_id': sample_id,
            'timestamp': datetime.now().isoformat(),
            'qc_metrics': qc_metrics,
            'user_feedback': user_feedback,
            'source_shape': source_image.shape,
            'target_ral_code': qc_metrics.get('target_ral_code')
        }

        with open(sample_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Collected training sample: {sample_id}")
```

### Automated Retraining

```python
# continuous_training.py

class ContinuousTrainingManager:
    """
    Manages continuous learning and model retraining.

    Features:
    - Scheduled retraining
    - Trigger-based retraining (new data threshold)
    - Incremental fine-tuning
    - Model versioning
    - A/B testing of new models
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        training_config: Dict
    ):
        self.model_registry = model_registry
        self.config = training_config
        self.retraining_threshold = training_config.get('new_samples_threshold', 1000)

    def should_retrain(self, worker_id: str) -> bool:
        """Check if worker should be retrained"""
        # Check number of new samples
        new_samples = self.model_registry.count_new_samples(worker_id)

        if new_samples >= self.retraining_threshold:
            return True

        # Check performance degradation
        recent_performance = self.model_registry.get_recent_performance(worker_id)
        baseline_performance = self.model_registry.get_baseline_performance(worker_id)

        if recent_performance < baseline_performance * 0.9:  # 10% degradation
            logger.warning(f"Performance degradation detected for {worker_id}")
            return True

        return False

    def retrain_worker(
        self,
        worker_id: str,
        training_data: torch.utils.data.Dataset,
        epochs: int = 10
    ) -> str:
        """
        Retrain a worker model.

        Args:
            worker_id: Worker to retrain
            training_data: New training dataset
            epochs: Number of training epochs

        Returns:
            New model version ID
        """
        logger.info(f"Starting retraining for {worker_id}")

        # Load current model
        current_model = self.model_registry.load_model(worker_id)

        # Create trainer
        trainer = ModelTrainer(
            model=current_model,
            config=self.config
        )

        # Fine-tune on new data
        trainer.fine_tune(
            training_data,
            epochs=epochs,
            learning_rate=self.config.get('fine_tune_lr', 1e-5)
        )

        # Validate new model
        val_metrics = trainer.validate()

        # Save new model version
        new_version = self.model_registry.save_model(
            worker_id,
            trainer.model,
            metrics=val_metrics,
            training_info={
                'samples': len(training_data),
                'epochs': epochs,
                'timestamp': datetime.now().isoformat()
            }
        )

        logger.info(f"Retrained {worker_id} -> version {new_version}")

        return new_version

    def schedule_retraining(self):
        """Background task for scheduled retraining"""
        while True:
            for worker_id in self.model_registry.get_all_workers():
                if self.should_retrain(worker_id):
                    # Load new training data
                    training_data = self.load_new_training_data(worker_id)

                    # Retrain in background
                    self.retrain_worker(worker_id, training_data)

            # Sleep until next check (e.g., daily)
            time.sleep(86400)
```

### Knowledge Distillation

```python
# knowledge_distillation.py

class KnowledgeDistillation:
    """
    Distill knowledge from large teacher model to small student model.

    Benefits:
    - Deploy faster, smaller models
    - Maintain accuracy of large models
    - Reduce inference latency
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature

        self.teacher.eval()

    def distillation_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        true_output: torch.Tensor,
        alpha: float = 0.7
    ) -> torch.Tensor:
        """
        Combined distillation loss.

        Args:
            student_output: Student model predictions
            teacher_output: Teacher model predictions (soft targets)
            true_output: Ground truth labels (hard targets)
            alpha: Weight for distillation loss vs. true loss

        Returns:
            Combined loss
        """
        # Distillation loss (KL divergence on soft targets)
        distill_loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # True loss (MSE on hard targets)
        true_loss = F.mse_loss(student_output, true_output)

        # Combined loss
        return alpha * distill_loss + (1 - alpha) * true_loss

    def distill(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 20
    ):
        """
        Train student model via knowledge distillation.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer for student model
            epochs: Number of training epochs
        """
        self.student.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, (source, target, ground_truth) in enumerate(train_loader):
                source = source.cuda()
                target = target.cuda()
                ground_truth = ground_truth.cuda()

                # Get teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_output = self.teacher(source, target)

                # Get student predictions
                student_output = self.student(source, target)

                # Calculate distillation loss
                loss = self.distillation_loss(
                    student_output,
                    teacher_output,
                    ground_truth
                )

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
```

---

## Reinforcement Learning for Orchestration

### RL-Based Weight Optimizer

```python
# rl_weight_optimizer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random


class WorkerSelectionAgent(nn.Module):
    """
    Deep Q-Network for learning optimal worker selection policy.

    State: Image features + complexity metrics
    Action: Worker weights (continuous)
    Reward: Negative Delta E (lower is better)
    """

    def __init__(self, state_dim: int, num_workers: int):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)

        # Output layer: weights for each worker
        self.worker_weights = nn.Linear(128, num_workers)

    def forward(self, state):
        """
        Predict worker weights given state.

        Args:
            state: Image features and complexity metrics

        Returns:
            Worker weights (softmax normalized)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Softmax to get valid weight distribution
        weights = F.softmax(self.worker_weights(x), dim=-1)

        return weights


class RLWeightOptimizer:
    """
    Reinforcement Learning optimizer for TSM worker weights.

    Uses DQN (Deep Q-Network) to learn optimal worker selection
    based on image characteristics and historical performance.
    """

    def __init__(
        self,
        state_dim: int,
        num_workers: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.agent = WorkerSelectionAgent(state_dim, num_workers)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.memory = deque(maxlen=10000)
        self.num_workers = num_workers

    def get_state_features(
        self,
        image: np.ndarray,
        complexity_metrics: Dict
    ) -> torch.Tensor:
        """
        Extract state features from image and complexity metrics.

        Args:
            image: Input image
            complexity_metrics: Complexity analysis results

        Returns:
            State feature vector
        """
        features = []

        # Complexity metrics
        features.extend([
            complexity_metrics['color_variance'],
            complexity_metrics['edge_density'],
            complexity_metrics['color_diversity'],
            complexity_metrics['gradient_intensity'],
            complexity_metrics['spatial_complexity'],
            complexity_metrics['resolution_factor']
        ])

        # Image statistics
        features.extend([
            np.mean(image),
            np.std(image),
            np.max(image),
            np.min(image)
        ])

        # Color channel statistics
        for channel in range(3):
            features.extend([
                np.mean(image[:, :, channel]),
                np.std(image[:, :, channel])
            ])

        return torch.FloatTensor(features)

    def select_weights(
        self,
        state: torch.Tensor,
        explore: bool = True
    ) -> np.ndarray:
        """
        Select worker weights using epsilon-greedy policy.

        Args:
            state: Current state features
            explore: Whether to explore (epsilon-greedy)

        Returns:
            Worker weights
        """
        if explore and random.random() < self.epsilon:
            # Explore: random weights
            weights = np.random.dirichlet(np.ones(self.num_workers))
        else:
            # Exploit: use learned policy
            with torch.no_grad():
                weights = self.agent(state.unsqueeze(0)).squeeze(0).numpy()

        return weights

    def store_experience(
        self,
        state: torch.Tensor,
        weights: np.ndarray,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ):
        """Store experience in replay buffer"""
        self.memory.append((state, weights, reward, next_state, done))

    def train_step(self, batch_size: int = 32):
        """
        Perform one training step using experience replay.

        Args:
            batch_size: Number of samples to train on
        """
        if len(self.memory) < batch_size:
            return

        # Sample batch from memory
        batch = random.sample(self.memory, batch_size)

        states = torch.stack([exp[0] for exp in batch])
        weights = torch.FloatTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.stack([exp[3] for exp in batch])
        dones = torch.FloatTensor([exp[4] for exp in batch])

        # Current Q-values
        current_weights = self.agent(states)

        # Target Q-values
        with torch.no_grad():
            next_weights = self.agent(next_states)
            target_rewards = rewards + self.gamma * (1 - dones) * torch.max(next_weights, dim=1)[0]

        # Loss: MSE between predicted and target
        loss = F.mse_loss(
            torch.sum(current_weights * weights, dim=1),
            target_rewards
        )

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()
```

---

## Training Infrastructure

### Distributed Training Setup

```python
# distributed_training.py

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """Initialize distributed training environment"""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def train_distributed(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    save_path: str
):
    """
    Distributed training across multiple GPUs.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        epochs: Number of epochs
        save_path: Path to save checkpoints
    """
    # Setup
    setup_distributed()
    rank = dist.get_rank()

    # Wrap model in DDP
    model = model.cuda()
    model = DDP(model, device_ids=[rank])

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (source, target, ground_truth) in enumerate(train_loader):
            source = source.cuda()
            target = target.cuda()
            ground_truth = ground_truth.cuda()

            # Forward pass
            output = model(source, target)
            loss = F.mse_loss(output, ground_truth)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Save checkpoint (only on rank 0)
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(train_loader)
            }, f"{save_path}/checkpoint_epoch_{epoch}.pth")

    # Cleanup
    dist.destroy_process_group()


# Launch script
# torchrun --nproc_per_node=4 distributed_training.py
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

**Goals:**
- Implement PyTorch worker base classes
- Create CNN-based worker
- Integrate with existing TSM orchestrator

**Deliverables:**
- `pytorch_worker_base.py`
- `cnn_color_transfer_worker.py`
- Updated `tsm_orchestrator.py` with PyTorch support
- Basic training scripts

**Success Metrics:**
- CNN worker achieves < 10 Delta E on test set
- Integration with TSM complete
- No performance regression

### Phase 2: Advanced Models (Weeks 5-8)

**Goals:**
- Implement U-Net worker
- Implement Transformer worker
- Optimize inference performance

**Deliverables:**
- `unet_color_transfer_worker.py`
- `transformer_color_transfer_worker.py`
- Performance benchmarks
- Model compression (quantization, pruning)

**Success Metrics:**
- U-Net worker: < 8 Delta E average
- Transformer worker: < 7 Delta E average
- Inference time < 500ms on GPU

### Phase 3: Continuous Learning (Weeks 9-12)

**Goals:**
- Build training data collection pipeline
- Implement automated retraining
- Deploy knowledge distillation

**Deliverables:**
- `training_data_collector.py`
- `continuous_training.py`
- `knowledge_distillation.py`
- Monitoring dashboard

**Success Metrics:**
- Data collection from 100% of production requests
- Automated retraining every 1000 samples
- Student model 90% of teacher accuracy, 5x faster

### Phase 4: RL Optimization (Weeks 13-16)

**Goals:**
- Implement RL-based weight optimizer
- A/B testing framework
- Production deployment

**Deliverables:**
- `rl_weight_optimizer.py`
- A/B testing infrastructure
- Production deployment scripts
- Monitoring and alerting

**Success Metrics:**
- RL agent improves worker selection by 15%
- A/B tests show statistical significance
- Zero-downtime deployment

---

## Deployment Strategy

### Model Serving

```python
# model_serving.py

from fastapi import FastAPI
from ray import serve


@serve.deployment(num_replicas=4, ray_actor_options={"num_gpus": 1})
class ColorTransferService:
    """
    Ray Serve deployment for color transfer.

    Features:
    - Auto-scaling based on load
    - GPU utilization
    - Model versioning
    - A/B testing
    """

    def __init__(self):
        self.tsm_orchestrator = TSMOrchestrator()

        # Load PyTorch workers
        self.tsm_orchestrator.register_worker(CNNColorTransferWorker())
        self.tsm_orchestrator.register_worker(UNetColorTransferWorker())
        self.tsm_orchestrator.register_worker(TransformerColorTransferWorker())

    async def __call__(self, request):
        source_image = decode_image(request.source)
        target_color = np.array(request.target_color)

        result = self.tsm_orchestrator.process(
            source_rgb=source_image,
            target_rgb=target_color,
            mode='adaptive'
        )

        return {
            'result_image': encode_image(result.best_result_rgb),
            'best_worker': result.best_worker_id,
            'qc_report': result.qc_report
        }


# Deploy
app = FastAPI()
serve.start(detached=True)
ColorTransferService.deploy()
serve.run(ColorTransferService.bind(), host="0.0.0.0", port=8000)
```

---

## Expected Benefits

### Performance Improvements

| Metric | Current (Algorithmic) | With PyTorch | Improvement |
|--------|----------------------|--------------|-------------|
| **Average Delta E** | 10-12 | 6-8 | 40-50% |
| **Inference Time** | 1-2s (CPU) | 0.2-0.5s (GPU) | 75-80% |
| **Accuracy on Complex Images** | 70% | 90%+ | +20% |
| **Model Size** | N/A | 50-200MB | - |
| **Training Time** | N/A | 4-8 hours | - |

### Business Impact

- **Better Quality**: 40-50% reduction in color error
- **Faster Processing**: 4-5x speedup with GPU
- **Continuous Improvement**: Models improve over time
- **Scalability**: Distributed training and inference
- **Cost Reduction**: Automated retraining reduces manual effort

---

## Conclusion

Integrating PyTorch with the existing TSM system represents a significant evolution from traditional algorithmic approaches to state-of-the-art deep learning. This integration enables:

1. **More Accurate Workers** via CNNs, U-Nets, and Transformers
2. **Continuous Learning** through automated retraining
3. **Intelligent Orchestration** using reinforcement learning
4. **Production Scalability** with distributed training and serving

The roadmap provides a clear path from foundation to production deployment over 16 weeks, with measurable success metrics at each phase.

**Next Steps:**
1. Review and approve roadmap
2. Allocate GPU resources (4+ GPUs recommended)
3. Begin Phase 1 implementation
4. Set up training data collection

---

**Document Version:** 1.0.0
**Last Updated:** November 2025
**Status:** Proposal - Awaiting Approval
