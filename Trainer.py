import torch
import torch.optim as optim
#!/usr/bin/env python3
import os
import process_data as prep
import Diffuser as diff
import unet
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.optim import Adam

save_path = './models/dif_model.pth'
os.makedirs('./models', exist_ok=True)
data, data_loader, test_Dloader = prep.get_and_load_dataset()
    



class Trainer:
    def __init__(self, model, diffuser, data_loader, epochs=10, lr=1e-4, device="cpu"):
        """
        Initializes the Trainer.
        
        Args:
            model: The neural network to train.
            diffuser: An instance of the Diffuser class.
            data_loader: PyTorch DataLoader for training data.
            epochs: Number of epochs to train for.
            lr: Learning rate for the optimizer.
            device: Device to train on ("cpu" or "cuda").
        """
        self.model = model.to(device)
        self.diffuser = diffuser
        self.data_loader = data_loader
        self.epochs = epochs
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.lr = lr

    def train(self):
        """
        Executes the training loop.
        """
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
          
            for step, batch in enumerate(data_loader):
                # Move batch to device
                batch = batch.to(self.device)
                Finput = batch[:, :3, :, :]  # First 3 channels
                Fpred = batch[:, 3:, :, :]  # Remaining 3 channels
                
                # Randomly sample timesteps
                t = torch.randint(0, self.diffuser.T, (batch.size(0),), device=self.device).long()

                # Zero gradients
                self.optimizer.zero_grad()

                # Compute loss
                loss = self.diffuser.get_loss(model = self.model,x_0 = Fpred, t =t , condition= Finput)

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                # Track loss
                epoch_loss += loss.item()
            
            # Log epoch loss
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss / len(self.data_loader)}")
            if epoch > 1 and epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, save_path)
                print(f"Model saved to {save_path}")
