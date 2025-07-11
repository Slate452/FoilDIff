import os
from copy import deepcopy
from collections import OrderedDict
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import Process_opf_data as prep
import datetime

def Error_log(model_name, error_message):
    """
    Log error message with timestamp.

    Args:
        model_name (str): Name of the model.
        error_message (str): Error message or exception trace.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = "error_log.txt"
    
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] Error in model: {model_name}\n")
        f.write(f"Message: {error_message}\n")
        f.write("=" * 80 + "\n")
    
    print(f"Error logged to {log_file}")

def Checkpoint_save(model, loss, l_epoch, save_path):
    """
    Save model checkpoint and training logs.

    Args:
        model (torch.nn.Module): The model to save.
        loss (float): The training loss at current epoch.
        l_epoch (int): The current epoch number.
        save_path (str): The full path to save the model file (should end in '.pth').
    """
    # Create base directory if it doesn't exist
    base_dir = os.path.dirname(save_path)
    os.makedirs(base_dir, exist_ok=True)

    # Save loss to CSV file
    loss_file = os.path.join(base_dir, 'loss.csv')
    with open(loss_file, 'a') as f:
        f.write(f"{l_epoch},{loss:.6f}\n")

    # Save model weights
    model_name = model.__class__.__name__
    model_dir = os.path.join(base_dir, f"{model_name}_epoch_{l_epoch}")
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_file = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Model checkpoint saved to {checkpoint_file}")

    # Save model architecture (optional)
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
    def __init__(self, model, diffuser, data_loader, epochs=10000, lr=1e-4, ema_decay=0.9999, device="cuda",save_path=None):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.ema_model = deepcopy(model).to(self.device)
        self.ema_model.eval()
        self.save_path = save_path if save_path else './model_checkpoint.pth'
        

        self.diffuser = diffuser
        self.data_loader = data_loader
        self.epochs = epochs
        self.lr = lr
        self.global_step = 0
        self.total_steps = self.epochs * len(self.data_loader)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.ema_decay = ema_decay
        self.progress_bar = tqdm(total=self.total_steps, desc="Training", dynamic_ncols=True)

    def train(self):
        self.model.train()
        loss_history = []
        plt.ion()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(self.data_loader):
                batch = batch.to(self.device)
                t = torch.randint(0, self.diffuser.T, (batch.size(0),), device=self.device).long()

                self.optimizer.zero_grad()
                loss = self.diffuser.get_loss(model=self.model, x_0=batch, t=t)
                loss.backward()
                self.optimizer.step()

                update_ema(self.ema_model, self.model, decay=self.ema_decay)

                epoch_loss += loss.item()
                self.progress_bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])
                self.progress_bar.update(1)

                batch = batch.cpu()
                torch.cuda.empty_cache()

            loss_history.append(epoch_loss / len(self.data_loader))

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
            plt.pause(0.01)

            if epoch % 100 == 0:
                Checkpoint_save(self.model, loss_history[-1], epoch, self.save_path)
                print(f"Checkpoint saved at epoch {epoch}")

        print("Training complete.")
        plt.savefig("training_loss_curve.png")

        return {
            "ema_model": self.ema_model,
            "last_model": self.model
        }


if __name__ == "__main__":
    save_path = './content/Feb2025_LAIL/models/dif_model.pth'
    os.makedirs(save_path, exist_ok=True)

    data, data_loader, test_Dloader = prep.get_and_load_dataset()

    model = ...      # define your model
    diffuser = ...   # define your diffuser

    trainer = Trainer(model, diffuser, data_loader, epochs=100)
    trained_model = trainer.train()
