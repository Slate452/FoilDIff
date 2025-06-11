
#!/usr/bin/env python3
import os
from copy import deepcopy
from glob import glob
from time import time
import argparse

import logging
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import process_data as prep


import torch
import torch.distributed
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.optim import Adam
from collections import OrderedDict

save_path = './content/Feb2025_LAIL/models/dif_model.pth'
os.makedirs(save_path, exist_ok=True)
data, data_loader, test_Dloader = prep.get_and_load_dataset()


def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger



class Trainer:
    def __init__(self, model, diffuser, data_loader, epochs=10000, lr=1e-4, device="cuda"):
        """
        Initializes the Trainer.
        
        Args:
            model: The neural network to train.
            diffuser: An instance of the Diffuser class.
            data_loader: PyTorch DataLoader for training data.
            epochs: Number of epochs to train for.
            lr: Learning rate for the optimizer.
            device: Device to train on ("cuda").
        """
        self.model = model.to(device)
        self.diffuser = diffuser
        self.data_loader = data_loader
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.global_step = 0
        self.total_steps = self.epochs * len(self.data_loader)
        self.progress_bar = tqdm(total=self.total_steps, desc="Training", dynamic_ncols=True)



    def train(self):
        """
        Executes the training loop.
        """
        self.model.train()
        loss_history = []
        plt.ion()  # interactive mode ON

        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
          
            for step, batch in enumerate(self.data_loader):
                # Move batch to device
                batch = batch.to(self.device)
                # Randomly sample timesteps
                t = torch.randint(0, self.diffuser.T, (batch.size(0),), device=self.device).long()

                # Zero gradients
                self.optimizer.zero_grad()
                # Compute loss
                loss = self.diffuser.get_loss(model = self.model,x_0 = batch, t =t)

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                
                # Track loss
                epoch_loss += loss.item()
                # Update progress bar
                self.progress_bar.set_postfix(loss=loss.item(), lr=self.scheduler.get_last_lr()[0])
                self.progress_bar.update(1)


                batch = batch.to("cpu")  # Moves the batch back to CPU
                torch.cuda.empty_cache()  # Clears unused memory in CUDA (optional)

            self.scheduler.step()
            # Log epoch loss
            loss_history.append(epoch_loss / len(self.data_loader))
            #
            clear_output(wait=True)
            plt.figure(figsize=(8, 4))
            plt.plot(loss_history, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Live Training Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.pause(0.01)  # allow the plot to refresh

            #print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss / len(self.data_loader)}")
        print("Training complete.")
        plt.savefig("training_loss_curve.png")
        return self.model
        
