# %%
import torch.nn as nn
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
from pytorch_lightning.loggers import WandbLogger
import torchvision.transforms as transforms

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        # Convert kernel_size to proper format if needed
        kernel_size = self._normalize_kernel_size(kernel_size)
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            List of tuples (h, c) for each layer. If None, initialize with zeros.

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            # Use provided hidden state
            if len(hidden_state) != self.num_layers:
                raise ValueError(f"Expected {self.num_layers} hidden states, got {len(hidden_state)}")
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _normalize_kernel_size(kernel_size):
        """Convert kernel_size to proper tuple format if it's a list"""
        if isinstance(kernel_size, list) and len(kernel_size) == 2:
            return tuple(kernel_size)
        return kernel_size

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError(f'`kernel_size` must be tuple or list of tuples, got {type(kernel_size)}: {kernel_size}')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# %%
class MovingMNISTDataset(Dataset):
    """Moving MNIST Dataset"""
    
    def __init__(self, root='./data', train=True, download=True):
        self.root = root
        self.train = train
        
        if download:
            self.download()
        
        # Load the same file for both train and test, but split differently
        self.data = np.load(os.path.join(root, 'mnist_test_seq.npy'))
        
        # Split the data: use first 8000 sequences for train, rest for test
        if train:
            self.data = self.data[:, :8000, :, :]
        else:
            self.data = self.data[:, 8000:, :, :]
    
    def download(self):
        os.makedirs(self.root, exist_ok=True)
        url = 'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy'
        filepath = os.path.join(self.root, 'mnist_test_seq.npy')
        
        if not os.path.exists(filepath):
            print('Downloading Moving MNIST dataset...')
            urllib.request.urlretrieve(url, filepath)
            print('Download completed!')
    
    def __len__(self):
        return self.data.shape[1]
    
    def __getitem__(self, idx):
        sequence = self.data[:, idx, :, :].astype(np.float32) / 255.0
        sequence = np.expand_dims(sequence, axis=1)  # Add channel dim
        
        input_seq = sequence[:10]   # First 10 frames
        target_seq = sequence[10:]  # Last 10 frames
        
        return torch.tensor(input_seq), torch.tensor(target_seq)


class ConvLSTMPredictor(pl.LightningModule):
    """PyTorch Lightning ConvLSTM for Moving MNIST with Weights & Biases integration"""
    
    def __init__(self, 
                 input_dim=1, 
                 hidden_dims=[64, 64, 64], 
                 kernel_size=(3, 3), 
                 num_layers=3,
                 learning_rate=1e-3,
                 batch_size=32,
                 log_images=True,
                 log_frequency=100):
        super().__init__()
        self.save_hyperparameters()
        
        # Convert kernel_size to tuple if it's a list (wandb config issue)
        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)
        
        # Encoder ConvLSTM
        self.encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dims,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=True
        )
        
        # Decoder ConvLSTM
        self.decoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dims,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=True
        )
        
        # Output layer
        self.output_conv = nn.Conv2d(
            in_channels=hidden_dims[-1],
            out_channels=input_dim,
            kernel_size=1
        )
        
        self.criterion = nn.MSELoss()
        self.log_images = log_images
        self.log_frequency = log_frequency
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, x, future_steps=10):
        # Encode input sequence
        _, encoder_states = self.encoder(x)
        
        # Use encoder states to initialize decoder
        decoder_hidden = encoder_states
        predictions = []
        
        # Use last input frame as initial decoder input
        decoder_input = x[:, -1:, :, :, :]
        
        for step in range(future_steps):
            # Forward through decoder with current hidden state
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Generate prediction from decoder output
            pred_frame = torch.sigmoid(self.output_conv(decoder_output[-1][:, -1, :, :, :]))
            predictions.append(pred_frame.unsqueeze(1))
            
            # Use prediction as next decoder input
            decoder_input = pred_frame.unsqueeze(1)
        
        return torch.cat(predictions, dim=1)
    
    def training_step(self, batch, batch_idx):
        input_seq, target_seq = batch
        predictions = self(input_seq, future_steps=10)
        loss = self.criterion(predictions, target_seq)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/mse', loss, on_step=True, on_epoch=True)
        
        # Log current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', current_lr, on_step=True, on_epoch=False)
        
        # Calculate additional metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(predictions - target_seq))
            self.log('train/mae', mae, on_step=True, on_epoch=True)
            
            # SSIM-like metric (simplified)
            ssim_loss = 1 - torch.mean((predictions * target_seq) / (torch.sqrt(predictions**2 + 1e-8) * torch.sqrt(target_seq**2 + 1e-8)))
            self.log('train/ssim_loss', ssim_loss, on_step=True, on_epoch=True)
        
        # Store outputs for logging images
        if self.log_images and batch_idx % self.log_frequency == 0:
            self.training_step_outputs.append({
                'input_seq': input_seq[:4].detach().cpu(),  # Log first 4 samples
                'target_seq': target_seq[:4].detach().cpu(),
                'predictions': predictions[:4].detach().cpu(),
                'loss': loss.detach().cpu(),
                'batch_idx': batch_idx
            })
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_seq, target_seq = batch
        predictions = self(input_seq, future_steps=10)
        loss = self.criterion(predictions, target_seq)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mse', loss, on_step=False, on_epoch=True)
        
        # Calculate additional metrics
        mae = torch.mean(torch.abs(predictions - target_seq))
        self.log('val/mae', mae, on_step=False, on_epoch=True)
        
        # SSIM-like metric (simplified)
        ssim_loss = 1 - torch.mean((predictions * target_seq) / (torch.sqrt(predictions**2 + 1e-8) * torch.sqrt(target_seq**2 + 1e-8)))
        self.log('val/ssim_loss', ssim_loss, on_step=False, on_epoch=True)
        
        # Store outputs for logging images
        if self.log_images and batch_idx == 0:  # Log only first validation batch
            self.validation_step_outputs.append({
                'input_seq': input_seq[:4].detach().cpu(),
                'target_seq': target_seq[:4].detach().cpu(),
                'predictions': predictions[:4].detach().cpu(),
                'loss': loss.detach().cpu()
            })
        
        return loss
    
    def test_step(self, batch, batch_idx):
        input_seq, target_seq = batch
        predictions = self(input_seq, future_steps=10)
        loss = self.criterion(predictions, target_seq)
        
        # Log test metrics
        self.log('test/loss', loss)
        mae = torch.mean(torch.abs(predictions - target_seq))
        self.log('test/mae', mae)
        
        return loss
    
    def on_train_epoch_end(self):
        if self.log_images and self.training_step_outputs:
            self._log_images(self.training_step_outputs, 'train')
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        if self.log_images and self.validation_step_outputs:
            self._log_images(self.validation_step_outputs, 'val')
            self.validation_step_outputs.clear()
    
    def _log_images(self, outputs, stage):
        """Log images to wandb"""
        if not outputs:
            return
        
        # Take the first output batch
        output = outputs[0]
        input_seq = output['input_seq']
        target_seq = output['target_seq']
        predictions = output['predictions']
        
        # Create visualizations
        images = []
        for i in range(min(2, input_seq.shape[0])):  # Log first 2 samples
            fig, axes = plt.subplots(3, 10, figsize=(20, 6))
            
            # Plot input sequence
            for t in range(10):
                if t < input_seq.shape[1]:
                    axes[0, t].imshow(input_seq[i, t, 0].numpy(), cmap='gray', vmin=0, vmax=1)
                    axes[0, t].set_title(f'Input {t+1}', fontsize=8)
                else:
                    axes[0, t].axis('off')
            
            # Plot target sequence
            for t in range(10):
                axes[1, t].imshow(target_seq[i, t, 0].numpy(), cmap='gray', vmin=0, vmax=1)
                axes[1, t].set_title(f'Target {t+1}', fontsize=8)
            
            # Plot predictions
            for t in range(10):
                axes[2, t].imshow(predictions[i, t, 0].numpy(), cmap='gray', vmin=0, vmax=1)
                axes[2, t].set_title(f'Pred {t+1}', fontsize=8)
            
            # Remove axis ticks
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            plt.suptitle(f'{stage.capitalize()} Sample {i+1}: Input vs Target vs Prediction', fontsize=12)
            plt.tight_layout()
            
            # Convert to wandb image
            images.append(wandb.Image(fig, caption=f"{stage}_sample_{i+1}"))
            plt.close(fig)
        
        # Log to wandb
        if hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({
                f"{stage}/predictions": images,
                "epoch": self.current_epoch
            })
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def train_dataloader(self):
        dataset = MovingMNISTDataset(train=True, download=True)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)  # Changed to 0 for debugging
    
    def val_dataloader(self):
        dataset = MovingMNISTDataset(train=False, download=True)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)  # Changed to 0 for debugging
    
    def test_dataloader(self):
        return self.val_dataloader()


def train_model(config=None):
    """Train the ConvLSTM model using PyTorch Lightning with wandb logging"""
    
    # Initialize wandb
    wandb.init(
        project="convlstm-moving-mnist",
        config=config or {
            "input_dim": 1,
            "hidden_dims": [64, 64, 64],
            "kernel_size": [3, 3],  # Changed to list format for wandb compatibility
            "num_layers": 3,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "max_epochs": 50,
            "architecture": "ConvLSTM",
            "dataset": "Moving MNIST",
            "optimizer": "Adam",
            "scheduler": "StepLR"
        },
        tags=["convlstm", "video-prediction", "pytorch-lightning"]
    )
    
    # Initialize model with wandb config
    model = ConvLSTMPredictor(
        input_dim=wandb.config.input_dim,
        hidden_dims=wandb.config.hidden_dims,
        kernel_size=wandb.config.kernel_size,
        num_layers=wandb.config.num_layers,
        learning_rate=wandb.config.learning_rate,
        batch_size=wandb.config.batch_size,
        log_images=True,
        log_frequency=100
    )
    
    # Log model architecture
    wandb.watch(model, log_freq=100, log_graph=True)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath='checkpoints/',
        filename='convlstm-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Wandb Logger
    wandb_logger = WandbLogger(
        project="convlstm-moving-mnist",
        log_model="all",  # Log model checkpoints
        save_dir="./wandb_logs"
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=wandb.config.max_epochs,
        accelerator='auto',
        devices='auto',
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=50,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    trainer.fit(model)
    
    # Test
    trainer.test(model)
    
    # Log final metrics
    wandb.log({
        "final_train_loss": trainer.callback_metrics.get("train/loss_epoch", 0),
        "final_val_loss": trainer.callback_metrics.get("val/loss", 0),
        "best_val_loss": checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else 0
    })
    
    # Finish wandb run
    wandb.finish()
    
    return model, trainer

def get_config_by_id(config_id):
    """Map config_id to specific hyperparameter combinations"""
    
    # Define all possible values
    learning_rates = np.logspace(-5, -2, 10)  # 10 values between 1e-5 and 1e-2
    batch_sizes = [8, 16, 32]
    hidden_dims_options = [
        [32, 32, 32],
        [64, 64, 64], 
        [128, 64, 32],
        [64, 128, 64]
    ]
    num_layers_options = [2, 3, 4]
    kernel_sizes = [[3, 3], [5, 5], [7, 7]]
    
    # Create all combinations (you could also use random sampling)
    configs = []
    for lr in learning_rates[:4]:  # Limit to manageable number
        for bs in batch_sizes:
            for hd in hidden_dims_options:
                for nl in num_layers_options:
                    for ks in kernel_sizes:
                        configs.append({
                            "input_dim": 1,
                            "hidden_dims": hd,
                            "kernel_size": ks,
                            "num_layers": nl,
                            "learning_rate": lr,
                            "batch_size": bs,
                            "max_epochs": 20,
                            "architecture": "ConvLSTM",
                            "dataset": "Moving MNIST",
                            "optimizer": "Adam",
                            "scheduler": "StepLR"
                        })
    
    # Return the config for this specific ID
    if config_id <= len(configs):
        return configs[config_id - 1]  # SLURM arrays are 1-indexed
    else:
        # Fallback to random configuration
        return {
            "input_dim": 1,
            "hidden_dims": np.random.choice(hidden_dims_options),
            "kernel_size": np.random.choice(kernel_sizes),
            "num_layers": np.random.choice(num_layers_options),
            "learning_rate": np.random.choice(learning_rates),
            "batch_size": np.random.choice(batch_sizes),
            "max_epochs": 20,
            "architecture": "ConvLSTM",
            "dataset": "Moving MNIST",
            "optimizer": "Adam",
            "scheduler": "StepLR"
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_id', type=int, required=True)
    args = parser.parse_args()
    
    config = get_config_by_id(args.config_id)
    print(f"Running configuration {args.config_id}: {config}")
    
    # Train with this specific configuration
    model, trainer = train_model(config)


