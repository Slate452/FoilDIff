#!/usr/bin/env python3
import os
import process_opf_data as prep
from process_opf_data import IMG_SIZE, BATCH_SIZE
import Diffuser as diff
import unet
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.optim import Adam
#from Trainer import *
import Transformer
import backbone

save_path = './models/dif_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('./models', exist_ok=True)

#data, data_loader, test_Dloader = prep.get_and_load_dataset()
UNet = unet.UNetWithAttention().to(device)

model = UNet # Choose between UNet or Transformer     
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100 # Try more!



def test_unet():
    noise_steps = 1000
    model = unet.UNetWithAttention(noise_steps=noise_steps,time_dim=256, depth= 4).to(device)
    r = torch.randint(0, noise_steps, (1,), dtype=torch.long)
    noisy_x = torch.randn(1, 3, 128, 128).to(device=device)  # Example input tensor
    Condition = torch.randn(1,3,128,128).to(device)  # (1, 3, 32, 32)

    output = model(x=noisy_x, t=r, c = Condition)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Testing complete.  Output shape:", output.shape)

def test_Transformer():
    # Parameters
    batch_size = 1
    in_channels = 3   # RGB
    image_size = 32
    num_classes = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model with 3 input channels
    model = Transformer.Transformer_L_8(in_channels=in_channels, num_classes=num_classes, learn_sigma = False).to(device)
    model.eval()

    # Create dummy RGB input
    x = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,), dtype=torch.long).to(device)       # (1,)
    y = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 3, 32, 32)

    # Forward pass
    with torch.no_grad():
        out = model(x, t, y)

    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {t.shape}")
    print(f"Label shape: {y.shape}")
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Output shape: {out.shape}")  # Expected: (1, out_channels, 32, 32)

def test_unet_with_dit():
    # Parameters
    batch_size = 1
    in_channels = 3       # e.g., 2 concatenated RGB images or multi-variate field
    image_size = 128
    noise_steps = 1000
   
    # Initialize model
    model = unet.UNetWithTransformer(noise_steps=noise_steps, time_dim=256, size=image_size).to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 6, 32, 32)
    t = torch.randint(0, noise_steps, (batch_size,), dtype=torch.long).to(device)  # (1,)
    c = torch.randn(1, 3, 128, 128)  # Example condition (flow parameters, e.g., 3 channels)

    # Forward pass
    with torch.no_grad():
        out = model(x, t , c)

    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {t.shape}")
    print(f"Condition shape: {c.shape}")
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Output shape: {out.shape}")  # Expected: (1, 6, 32, 32)


def test_UViT():
    # Parameters
    batch_size = 1
    in_channels = 3   # RGB
    image_size = 32
    num_classes = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model with 3 input channels
    model = unet.UViT(in_channels=in_channels, learn_sigma = False).to(device)
    model.eval()

    # Create dummy RGB input
    x = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,), dtype=torch.long).to(device)       # (1,)
    y = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 3, 32, 32)

    # Forward pass
    with torch.no_grad():
        out = model(x, t, y)

    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {t.shape}")
    print(f"Label shape: {y.shape}")
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Output shape: {out.shape}")  # Expected: (1, out_channels, 32, 32)

def test_unet_with_uvit():
    # Parameters
    batch_size = 1
    in_channels = 3       # e.g., 2 concatenated RGB images or multi-variate field
    image_size = 128
    noise_steps = 1000
   
    # Initialize model
    model = unet.UNetwithUViT(noise_steps=noise_steps, time_dim=256, size=image_size).to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 6, 32, 32)
    t = torch.randint(0, noise_steps, (batch_size,), dtype=torch.long).to(device)  # (1,)
    c = torch.randn(1, 3, 128, 128)  # Example condition (flow parameters, e.g., 3 channels)

    # Forward pass
    with torch.no_grad():
        out = model(x, t , c)

    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {t.shape}")
    print(f"Condition shape: {c.shape}")
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Output shape: {out.shape}")  # Expected: (1, 6, 32, 32)


def sample_diffusion(technique="ddpm", timestep=300,skip = 5, plot = False):
    # Parameters
    batch_size = 1
    in_channels = 3       # e.g., 2 concatenated RGB images or multi-variate field
    image_size = 128
    noise_steps = timestep
    inputs = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 3, 32, 32)
    # Initialize model
    model = unet.UNetWithTransformer(noise_steps=noise_steps, time_dim=256, size =image_size).to(device)
    model.eval()

    diffuser = diff.Diffuser(timesteps=timestep, device=device, sample_trajectory_factor=skip)  # Adjust timesteps and device as needed
    prediction=diffuser.sample_from_noise(model,inputs,Tech = technique)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {prediction.shape}")  # Expected: (1, 3, 32, 32)
    if plot:
        prep.plot(prediction)
    return prediction

'''  
def get_single_input():
    i = data[0].to(device)
    Finput = i[:3, :, :]  # First 3 channels
    #Fpred = batch[:, 3:, :, :]  # Remaining 3 channels
    #prep.plot(i)
    return Finput
     
def Train(save_path= save_path) :   
    diffuser = diff.Diffuser(timesteps=300, device="cuda")  # Adjust timesteps and device as needed
    trainer = Trainer(model=model, diffuser=diffuser, data_loader=data_loader, epochs=150, lr=1e-4, device=device)
    # Start training
    trainer.train()
    # Save the model
    torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)      
    print(f"Model saved to {save_path}")
            

def load_model(model_path=save_path, device=device):
    model = unet.UNetWithAttention().to(device)  # Rebuild the model and move it to the appropriate device
    optimizer = Adam(model.parameters(), lr=1e-4)  # Recreate the optimizer

    # Load the state dictionaries
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model



def test_sample(device = device):
    Done = False
    model, optimizer = load_model()
    optimizer.zero_grad()
    diff.sample_plot_image(model,device)

    #smaple timestep
    #print 1 sample ]

'''

#test_unet()
#test_Transformer()
test_unet_with_dit()
#test_UViT()
#test_unet_with_uvit()
#sample_diffusion()