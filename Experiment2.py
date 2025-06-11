from pytorch_lightning import seed_everything
seed_everything(1, workers=True)

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rioxarray as rxr
import xarray as xr
import torchvision.transforms as transforms
from datetime import datetime
import re
from typing import List, Tuple, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import urllib.request
import wandb
import math
import argparse

class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim=5):
        super().__init__()
        self.embed_dim = embed_dim
    
    def forward(self, years, months, days):
        return self._sinusoidal_encoding(years, months, days)
    
    def _sinusoidal_encoding(self, years, months, days):        
        # Year encoding (linear trend)
        year_norm = (years - 1985) / 55.0
        
        # Month encoding (cyclical)
        month_rad = 2 * math.pi * (months - 1) / 12.0
        month_sin = torch.sin(month_rad)
        month_cos = torch.cos(month_rad)
        
        # Day encoding (cyclical within month)
        day_rad = 2 * math.pi * (days - 1) / 31.0
        day_sin = torch.sin(day_rad)
        day_cos = torch.cos(day_rad)
        
        # Combine features [batch_size, 5]
        temporal_features = torch.stack([
            year_norm, month_sin, month_cos, 
            day_sin, day_cos
        ], dim=-1)
        
        return temporal_features

class ConvLSTMCell(nn.Module):
    """ConvLSTM cell implementation."""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, 
                             kernel_size, padding=self.padding, bias=bias)
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i, f, o = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        h, w = image_size
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
                torch.zeros(batch_size, self.hidden_dim, h, w, device=device))

class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM."""
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * num_layers
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * num_layers
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        self.cell_list = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else self.hidden_dim[i-1], 
                        self.hidden_dim[i], self.kernel_size[i], bias)
            for i in range(num_layers)
        ])
    
    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, seq_len, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = [cell.init_hidden(b, (h, w)) for cell in self.cell_list]
        
        layer_output_list, last_state_list = [], []
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t], [h, c])
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        return layer_output_list, last_state_list

class TemporalFusionModule(nn.Module):
    """Module to fuse temporal encoding with spatial features."""
    def __init__(self, temporal_dim=5, spatial_channels=64, fused_channels=64):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.spatial_channels = spatial_channels
        self.fused_channels = fused_channels
        
        # Project temporal features to spatial dimensions
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_dim, spatial_channels),
            nn.ReLU(),
            nn.Linear(spatial_channels, spatial_channels)
        )
        
        # Fusion layer
        self.fusion_conv = nn.Conv2d(spatial_channels * 2, fused_channels, 1)
        
    def forward(self, spatial_features, temporal_features):
        # spatial_features: [batch, channels, height, width]
        # temporal_features: [batch, temporal_dim]
        
        batch_size, channels, height, width = spatial_features.shape
        
        # Project temporal features
        temporal_proj = self.temporal_proj(temporal_features)  # [batch, spatial_channels]
        
        # Expand temporal features to spatial dimensions
        temporal_spatial = temporal_proj.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, self.spatial_channels, height, width
        )
        
        # Concatenate and fuse
        combined = torch.cat([spatial_features, temporal_spatial], dim=1)
        fused = self.fusion_conv(combined)
        
        return fused

class SanAntonioSatelliteDataset(Dataset):
    """Enhanced dataset with temporal encoding support."""
    
    def __init__(self, data_dir: str, sequence_length: int = 5, target_length: int = 3,
                 image_size: int = 512, normalize: bool = True, temporal_stride: int = 1):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.target_length = target_length
        self.image_size = image_size
        self.normalize = normalize
        self.temporal_stride = temporal_stride
        
        # Get sorted .tif files by date
        self.tif_files = sorted(
            glob.glob(os.path.join(data_dir, "*.tif")),
            key=lambda f: datetime.strptime(re.search(r'(\d{4}-\d{2}-\d{2})', f).group(1), '%Y-%m-%d')
        )
        
        # Valid sequence starting indices
        total_needed = (sequence_length + target_length - 1) * temporal_stride + 1
        self.valid_sequences = list(range(len(self.tif_files) - total_needed + 1))
        
        print(f"Found {len(self.tif_files)} .tif files, {len(self.valid_sequences)} sequences")
    
    def _extract_date_from_filename(self, filename: str) -> Tuple[int, int, int]:
        """Extract year, month, day from filename."""
        # Extract date from San_Antonio_YYYY-MM-DD.tif format
        date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
        if date_match:
            year, month, day = map(int, date_match.groups())
            return year, month, day
        else:
            raise ValueError(f"Could not extract date from filename: {filename}")
    
    def _load_and_crop_tif(self, tif_path: str) -> torch.Tensor:
        """Load RGB channels from .tif and center crop to target size."""
        # Load with rioxarray (automatically handles CRS, transforms, etc.)
        da = rxr.open_rasterio(tif_path, chunks={'band': 1, 'x': 512, 'y': 512})
        
        # Take first 3 bands as RGB, center crop
        rgb = da.isel(band=slice(0, 3))
        h, w = rgb.sizes['y'], rgb.sizes['x']
        
        # Center crop indices
        center_y, center_x = h // 2, w // 2
        half_size = self.image_size // 2
        y_slice = slice(max(0, center_y - half_size), center_y + half_size)
        x_slice = slice(max(0, center_x - half_size), center_x + half_size)
        
        cropped = rgb.isel(y=y_slice, x=x_slice)
        
        # Convert to numpy and ensure correct shape/dtype
        data = cropped.values.astype(np.float32)
        
        # Pad if necessary to reach target size
        if data.shape[1] < self.image_size or data.shape[2] < self.image_size:
            padded = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
            h, w = data.shape[1], data.shape[2]
            start_h, start_w = (self.image_size - h) // 2, (self.image_size - w) // 2
            padded[:, start_h:start_h+h, start_w:start_w+w] = data
            data = padded
        
        # Normalize to [0,1]
        if self.normalize:
            if data.max() > 1:  # Assume uint8/uint16 if values > 1
                data = data / (65535.0 if data.max() > 255 else 255.0)
            else:
                # Percentile normalization for float data
                p1, p99 = np.percentile(data, [1, 99])
                if p99 > p1:
                    data = np.clip((data - p1) / (p99 - p1), 0, 1)
        
        return torch.from_numpy(data)
    
    def __len__(self) -> int:
        return len(self.valid_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start_idx = self.valid_sequences[idx]
        
        # Get file indices for input and target sequences
        input_indices = [start_idx + i * self.temporal_stride for i in range(self.sequence_length)]
        target_indices = [start_idx + (self.sequence_length + i) * self.temporal_stride 
                         for i in range(self.target_length)]
        
        # Load image sequences
        input_seq = torch.stack([self._load_and_crop_tif(self.tif_files[i]) for i in input_indices])
        target_seq = torch.stack([self._load_and_crop_tif(self.tif_files[i]) for i in target_indices])
        
        # Extract temporal information
        input_dates = []
        target_dates = []
        
        for i in input_indices:
            year, month, day = self._extract_date_from_filename(self.tif_files[i])
            input_dates.append([year, month, day])
            
        for i in target_indices:
            year, month, day = self._extract_date_from_filename(self.tif_files[i])
            target_dates.append([year, month, day])
        
        input_temporal = torch.tensor(input_dates, dtype=torch.float32)  # [seq_len, 3]
        target_temporal = torch.tensor(target_dates, dtype=torch.float32)  # [target_len, 3]
        
        return input_seq, target_seq, input_temporal, target_temporal

class SanAntonioDataModule(pl.LightningDataModule):
    """Lightning DataModule for San Antonio satellite data with temporal encoding."""
    
    def __init__(self, data_dir: str, sequence_length: int = 10, target_length: int = 5,
                 image_size: int = 512, batch_size: int = 4, num_workers: int = 4,
                 train_split: float = 0.8, val_split: float = 0.1, **kwargs):
        super().__init__()
        self.save_hyperparameters()
    
    def setup(self, stage: Optional[str] = None):
        dataset = SanAntonioSatelliteDataset(
            self.hparams.data_dir, 
            self.hparams.sequence_length, 
            self.hparams.target_length,
            self.hparams.image_size, 
            **{k: v for k, v in self.hparams.items() 
               if k not in ['data_dir', 'batch_size', 'num_workers', 'train_split', 'val_split', 
                           'sequence_length', 'target_length', 'image_size']}
        )
        
        # Split dataset
        total = len(dataset)
        train_size = int(self.hparams.train_split * total)
        val_size = int(self.hparams.val_split * total)
        
        self.train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        self.val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
        self.test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total))
        
        print(f"Splits - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def _dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=shuffle,
                         num_workers=self.hparams.num_workers, pin_memory=True,
                         persistent_workers=self.hparams.num_workers > 0)
    
    def train_dataloader(self): return self._dataloader(self.train_dataset, shuffle=True)
    def val_dataloader(self): return self._dataloader(self.val_dataset)
    def test_dataloader(self): return self._dataloader(self.test_dataset)

class SatelliteConvLSTMPredictor(pl.LightningModule):
    """Enhanced ConvLSTM model with temporal encoding for satellite imagery prediction."""
    
    def __init__(self, input_dim=3, hidden_dims=[64, 64, 64], kernel_size=(3, 3), 
                 num_layers=3, learning_rate=1e-3, target_length=5, batch_size=4,
                 temporal_dim=5, use_temporal_fusion=True,
                 log_images=True, log_frequency=100):
        super().__init__()
        self.save_hyperparameters()
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(embed_dim=temporal_dim)
        
        # Both encoder and decoder use same input dimensions
        self.encoder = ConvLSTM(input_dim, hidden_dims, kernel_size, num_layers, True, True)
        self.decoder = ConvLSTM(input_dim, hidden_dims, kernel_size, num_layers, True, True)
        
        # Temporal fusion modules
        if use_temporal_fusion:
            self.encoder_temporal_fusion = TemporalFusionModule(
                temporal_dim, hidden_dims[-1], hidden_dims[-1]
            )
            self.decoder_temporal_fusion = TemporalFusionModule(
                temporal_dim, hidden_dims[-1], hidden_dims[-1]
            )
        
        # Output projection
        self.output_conv = nn.Conv2d(hidden_dims[-1], input_dim, 1)
        self.criterion = nn.MSELoss()
        
        # For logging
        self.log_images = log_images
        self.log_frequency = log_frequency
        self.step_count = 0
    
    def forward(self, x, input_temporal, target_temporal):
        # x: [batch, seq_len, channels, height, width]
        # input_temporal: [batch, seq_len, 3] (year, month, day)
        # target_temporal: [batch, target_len, 3]
        
        batch_size, seq_len = x.shape[:2]
        target_len = target_temporal.shape[1]
        
        # Encode input sequence
        encoder_outputs, encoder_states = self.encoder(x)
        
        # Apply temporal fusion to encoder states if enabled
        if self.hparams.use_temporal_fusion:
            # Use last input temporal encoding for encoder fusion
            last_input_temporal = input_temporal[:, -1]  # [batch, 3]
            temporal_encoding = self.temporal_encoder(
                last_input_temporal[:, 0], 
                last_input_temporal[:, 1], 
                last_input_temporal[:, 2]
            )
            
            # Fuse with encoder output
            encoder_features = encoder_outputs[-1][:, -1]  # [batch, hidden_dim, H, W]
            fused_encoder = self.encoder_temporal_fusion(encoder_features, temporal_encoding)
            
            # Update encoder states
            encoder_states[-1][0] = fused_encoder
        
        # Decode target sequence
        predictions = []
        decoder_input = x[:, -1:, :, :, :]  # Start with last input frame
        decoder_hidden = encoder_states
        
        for t in range(target_len):
            # Get temporal encoding for current target timestep
            current_temporal = target_temporal[:, t]  # [batch, 3]
            temporal_encoding = self.temporal_encoder(
                current_temporal[:, 0], 
                current_temporal[:, 1], 
                current_temporal[:, 2]
            )
            
            # Decode one step
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Apply temporal fusion if enabled
            if self.hparams.use_temporal_fusion:
                decoder_features = decoder_output[-1][:, -1]  # [batch, hidden_dim, H, W]
                fused_decoder = self.decoder_temporal_fusion(decoder_features, temporal_encoding)
            else:
                fused_decoder = decoder_output[-1][:, -1]
            
            # Generate prediction
            pred_frame = torch.sigmoid(self.output_conv(fused_decoder))
            predictions.append(pred_frame.unsqueeze(1))
            
            # Use prediction as next input
            decoder_input = pred_frame.unsqueeze(1)
        
        return torch.cat(predictions, dim=1)
    
    def _step(self, batch, stage):
        input_seq, target_seq, input_temporal, target_temporal = batch
        predictions = self(input_seq.float(), input_temporal, target_temporal)
        loss = self.criterion(predictions, target_seq.float())
        
        # Enhanced logging with additional metrics
        self.log(f'{stage}/loss', loss, prog_bar=True, sync_dist=True, on_epoch=True, on_step=True)
        
        # Calculate additional metrics
        if stage in ['val', 'test']:
            with torch.no_grad():
                mae = torch.mean(torch.abs(predictions - target_seq.float()))
                mse = torch.mean((predictions - target_seq.float()) ** 2)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
                
                self.log(f'{stage}/mae', mae, sync_dist=True)
                self.log(f'{stage}/mse', mse, sync_dist=True)
                self.log(f'{stage}/psnr', psnr, sync_dist=True)
        
        # Log images periodically
        if self.log_images and self.step_count % self.log_frequency == 0 and stage == 'val':
            self._log_prediction_images(input_seq, target_seq, predictions, input_temporal, target_temporal, stage)
        
        self.step_count += 1
        return loss
    
    def _log_prediction_images(self, input_seq, target_seq, predictions, input_temporal, target_temporal, stage):
        """Log prediction visualizations with temporal information to wandb"""
        try:
            # Take first sample from batch
            input_sample = input_seq[0].detach().cpu().float().numpy()
            target_sample = target_seq[0].detach().cpu().float().numpy()
            pred_sample = predictions[0].detach().cpu().float().numpy()
            input_temp = input_temporal[0].detach().cpu().numpy()
            target_temp = target_temporal[0].detach().cpu().numpy()
            
            # Create visualization
            num_frames = min(5, self.hparams.target_length)
            fig, axes = plt.subplots(3, num_frames, figsize=(num_frames * 4, 12))
            
            if num_frames == 1:
                axes = axes.reshape(3, 1)
            
            for t in range(num_frames):
                # Last input frame (only show in first column)
                if t == 0:
                    input_img = np.transpose(input_sample[-1], (1, 2, 0)).astype(np.float32)
                    input_img = np.clip(input_img, 0, 1)
                    axes[0, t].imshow(input_img)
                    # Add temporal info
                    last_date = input_temp[-1]
                    axes[0, t].set_title(f'Last Input\n{int(last_date[0])}-{int(last_date[1]):02d}-{int(last_date[2]):02d}', 
                                        fontsize=10)
                else:
                    axes[0, t].axis('off')
                
                # Target frame
                target_img = np.transpose(target_sample[t], (1, 2, 0)).astype(np.float32)
                target_img = np.clip(target_img, 0, 1)
                axes[1, t].imshow(target_img)
                target_date = target_temp[t]
                axes[1, t].set_title(f'Target {t+1}\n{int(target_date[0])}-{int(target_date[1]):02d}-{int(target_date[2]):02d}', 
                                    fontsize=10)
                
                # Predicted frame
                pred_img = np.transpose(pred_sample[t], (1, 2, 0)).astype(np.float32)
                pred_img = np.clip(pred_img, 0, 1)
                axes[2, t].imshow(pred_img)
                axes[2, t].set_title(f'Predicted {t+1}\n{int(target_date[0])}-{int(target_date[1]):02d}-{int(target_date[2]):02d}', 
                                    fontsize=10)
                
                # Remove axes
                for i in range(3):
                    axes[i, t].set_xticks([])
                    axes[i, t].set_yticks([])
            
            # Add row labels
            axes[0, 0].set_ylabel('Input', fontsize=14, rotation=90, labelpad=20)
            axes[1, 0].set_ylabel('Target', fontsize=14, rotation=90, labelpad=20)
            axes[2, 0].set_ylabel('Predicted', fontsize=14, rotation=90, labelpad=20)
            
            plt.tight_layout()
            
            # Log to wandb
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({
                    f'{stage}/predictions_with_dates': wandb.Image(fig),
                    'epoch': self.current_epoch,
                    'step': self.step_count
                })
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Error logging images: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
    
    def training_step(self, batch, batch_idx): 
        return self._step(batch, 'train')
    
    def validation_step(self, batch, batch_idx): 
        return self._step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

def validate_config(config):
    """Validate sweep configuration parameters"""
    
    # Ensure target_length < sequence_length (need at least 1 gap)
    if config.target_length >= config.sequence_length:
        config.target_length = max(1, config.sequence_length - 2)
        wandb.log({"config_warning": f"Adjusted target_length to {config.target_length}"})
    
    # Ensure hidden_dims matches num_layers
    if len(config.hidden_dims) != config.num_layers:
        # If mismatch, skip this configuration
        print(f"Skipping config: hidden_dims length {len(config.hidden_dims)} != num_layers {config.num_layers}")
        wandb.log({"config_error": "hidden_dims_num_layers_mismatch", "status": "skipped"})
        wandb.finish()
        return None
    
    # Validate sequence/target combinations based on your original script logic
    valid_combinations = [
        (5, 3), (8, 3), (8, 5), (10, 5), (12, 5), (12, 8), 
        (15, 8), (15, 10), (16, 6), (16, 8), (16, 10)
    ]
    
    current_combo = (config.sequence_length, config.target_length)
    if current_combo not in valid_combinations:
        # Find closest valid combination
        valid_target_lengths = [tl for sl, tl in valid_combinations if sl == config.sequence_length]
        if valid_target_lengths:
            config.target_length = min(valid_target_lengths, key=lambda x: abs(x - config.target_length))
        else:
            # If no valid sequence length, find closest
            closest_seq = min([sl for sl, tl in valid_combinations], 
                            key=lambda x: abs(x - config.sequence_length))
            config.sequence_length = closest_seq
            config.target_length = min([tl for sl, tl in valid_combinations if sl == closest_seq])
        
        wandb.log({"config_warning": f"Adjusted to valid combination: seq={config.sequence_length}, target={config.target_length}"})
    
    # Memory efficiency check - skip configurations that might OOM
    memory_estimate = config.batch_size * config.image_size * config.image_size * max(config.hidden_dims)
    if memory_estimate > 500_000_000:  # Rough threshold
        print(f"Skipping config due to high memory estimate: {memory_estimate}")
        wandb.log({"config_error": "high_memory_estimate", "status": "skipped", "memory_estimate": memory_estimate})
        wandb.finish()
        return None
    
    # Architecture consistency check
    if config.num_layers == 2 and len([d for d in config.hidden_dims if d > 128]) > 0:
        if config.batch_size > 2:
            config.batch_size = 2
            wandb.log({"config_warning": f"Reduced batch_size to {config.batch_size} for large 2-layer architecture"})
    
    return config
def train_satellite_model_sweep():
    """Train function for WandB sweep - no config parameter needed"""
    
    # Initialize wandb - sweep will handle config
    wandb.init(
        project="convlstm-satellite-temporal-hp",
        tags=["convlstm", "temporal-encoding", "satellite-prediction", "bayesian-sweep"]
    )
    
    # Data module using wandb.config
    data_module = SanAntonioDataModule(
        data_dir=wandb.config.data_dir,
        sequence_length=wandb.config.sequence_length,
        target_length=wandb.config.target_length,
        image_size=wandb.config.image_size,
        batch_size=wandb.config.batch_size,
        num_workers=wandb.config.num_workers,
        train_split=wandb.config.train_split,
        val_split=wandb.config.val_split,
        temporal_stride=wandb.config.temporal_stride,
        normalize=wandb.config.normalize
    )
    
    # Initialize model with wandb config
    model = SatelliteConvLSTMPredictor(
        input_dim=wandb.config.input_dim,
        hidden_dims=wandb.config.hidden_dims,
        kernel_size=tuple(wandb.config.kernel_size),
        num_layers=wandb.config.num_layers,
        learning_rate=wandb.config.learning_rate,
        target_length=wandb.config.target_length,
        batch_size=wandb.config.batch_size,
        temporal_dim=wandb.config.temporal_dim,
        use_temporal_fusion=wandb.config.use_temporal_fusion,
        log_images=wandb.config.log_images,
        log_frequency=wandb.config.log_frequency
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=1
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Wandb Logger
    wandb_logger = WandbLogger(
        project="convlstm-satellite-temporal-hp",
        log_model=False  # Don't save models for sweep
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=wandb.config.max_epochs,
        accelerator='auto',
        devices=1,
        precision=wandb.config.precision,
        gradient_clip_val=wandb.config.gradient_clip_val,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=False,  # Disable for sweep
        enable_model_summary=False
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Log final metric for sweep optimization
    wandb.log({"final_val_loss": trainer.callback_metrics.get("val/loss", float('inf'))})

if __name__ == "__main__":
    # Remove the argument parser - sweep handles everything
    train_satellite_model_sweep()