#!/usr/bin/env python3
import os
import process_data as prep
from process_data import IMG_SIZE, BATCH_SIZE
import Diffuser as diff
import unet
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.optim import Adam
from Trainer import *

save_path = './models/dif_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('./models', exist_ok=True)

data, data_loader, test_Dloader = prep.get_and_load_dataset()
model = unet.UNetWithAttention()

model.to(device)       
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100 # Try more!
def get_single_input():
    i = data[0].to(device)
    print(i.shape)
    Finput = i[:3, :, :]  # First 3 channels
    #Fpred = batch[:, 3:, :, :]  # Remaining 3 channels
    #prep.plot(i)
    return Finput

def Train() :   
    diffuser = diff.Diffuser(timesteps=300, device="cuda")  # Adjust timesteps and device as needed
    trainer = Trainer(model=model, diffuser=diffuser, data_loader=data_loader, epochs=300, lr=1e-4, device=device)
    # Start training
    trainer.train()
    # Save the model
    torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)      
    print(f"Model saved to {save_path}")
            

def load_model(model_path=save_path, device=device):
    model = unet.build_unet().to(device)  # Rebuild the model and move it to the appropriate device
    optimizer = Adam(model.parameters(), lr=0.001)  # Recreate the optimizer

    # Load the state dictionaries
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

def sample_diffusion(model,inputs,num_sample=100):
    model.eval();model.to(device);predictions=[]
    diffuser = diff.Diffuser(timesteps=300, device="cuda")  # Adjust timesteps and device as needed
    prediction=diffuser.sample_from_noise(model,inputs,Tech = "ddpm")
    prep.plot(prediction)
    return prediction

def test_sample(device = device):
    Done = False
    model, optimizer = load_model()
    optimizer.zero_grad()
    diff.sample_plot_image(model,device)

    #smaple timestep
    #print 1 sample ]
# Assuming you have a PyTorch model `MyModel` and DataLoader `train_loader`

