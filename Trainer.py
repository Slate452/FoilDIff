import os
import datetime
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from IPython.display import clear_output
except ImportError:
    def clear_output(wait=False): pass  # fallback if not in notebook

import Process_opf_data as prep  # your data loader module
# from your_model_file import UViT, Diffuser  # <-- Make sure to import your model and diffuser here


def Error_log(model_name, error_message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = "error_log.txt"
    
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] Error in model: {model_name}\n")
        f.write(f"Message: {error_message}\n")
        f.write("=" * 80 + "\n")
    
    print(f"Error logged to {log_file}")


def Checkpoint_save(model, loss, l_epoch, save_path):
    base_dir = os.path.dirname(save_path)
    os.makedirs(base_dir, exist_ok=True)

    # Save loss to CSV
    loss_file = os.path.join(base_dir, 'loss.csv')
    with open(loss_file, 'a') as f:
        f.write(f"{l_epoch},{loss:.6f}\n")

    # Save model weights
    model_name = model.__class__.__name__
    model_dir = os.path.join(base_dir, f"{model_name}_epoch_{l_epoch}_loss_{loss:.4f}")
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_file = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Model checkpoint saved to {checkpoint_file}")

    # Save architecture
    arch_file = os.path.join(model_dir, "architecture.txt")
    with open(arch_file, 'w') as f:
        f.write(str(model))


def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class Trainer:
    def __init__(self, model, diffuser, data_loader, epochs=10000, lr=1e-4, ema_decay=0.9999, device="cuda", save_path=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.ema_model = deepcopy(model).to(self.device)
        self.ema_model.eval()

        self.diffuser = diffuser
        self.data_loader = data_loader
        self.epochs = epochs
        self.lr = lr
        self.ema_decay = ema_decay
        self.save_path = save_path if save_path else './checkpoints/model_checkpoint.pth'
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.total_steps = self.epochs * len(self.data_loader)
        self.progress_bar = tqdm(total=self.total_steps, desc="Training", dynamic_ncols=True)

    def train_step(self, model: torch.nn.Module, batch):
        batch = batch.to(self.device)  # shape: [B, 6, H, W]
        condition = batch[:, :3, :, :].to(self.device)  # shape: [B, 3, H, W]
        targets   = batch[:, 3:, :, :].to(self.device)  # shape: [B, 3, H, W]
        B = batch.size(0)
        t = torch.randint(0, self.diffuser.steps, (B,), dtype=torch.long).to(self.device)
        noise = torch.randn_like(targets).to(self.device)  # shape: [B, 3, H, W]
        xt = self.diffuser.forward_diffusion(targets, t, noise)

        predicted_noise = model(xt, t, condition)
        loss = F.mse_loss(predicted_noise, noise)

        del batch, condition, targets
        torch.cuda.empty_cache()
        return loss

    def train(self):
        try:
            self.model.train()
            loss_history = []
            plt.ion()

            for epoch in range(self.epochs):
                epoch_loss = 0.0

                for step, batch in enumerate(self.data_loader):
                    self.optimizer.zero_grad()
                    loss = self.train_step(self.model, batch)
                    loss.backward()
                    self.optimizer.step()

                    update_ema(self.ema_model, self.model, decay=self.ema_decay)
                    
                    epoch_loss += loss.item()
                    self.progress_bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])
                    self.progress_bar.update(1)

                    del batch , loss
                    torch.cuda.empty_cache()

                loss_history.append(epoch_loss / len(self.data_loader))
                self.scheduler.step()

                # Live plot
                plt.clf()
                plt.plot(loss_history, label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Progress')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.pause(0.01)

                # Save checkpoint
                if (epoch + 1) % 100 == 0:
                    Checkpoint_save(self.model, loss_history[-1], epoch + 1, self.save_path)

            plt.ioff()
            plt.savefig("training_loss_curve.png")
            print("Training complete.")

            return {
                "ema_model": self.ema_model,
                "last_model": self.model
            }

        except Exception as e:
            import traceback
            Error_log(self.model.__class__.__name__, traceback.format_exc())
            