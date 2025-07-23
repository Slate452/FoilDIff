#!/usr/bin/env python3
import os
import OpenFoam_pipeline as prep
from OpenFoam_pipeline import IMG_SIZE, BATCH_SIZE
import Diffuser as diff
import Backbone as backbone
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.optim import Adam
import Unifoil_pipeline as unifoil
#from Trainer import *
import Transformer

save_path = './models/dif_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('./models', exist_ok=True)

#data, data_loader, test_Dloader = prep.get_and_load_dataset()
UNet = backbone.UNetWithAttention().to(device)

model = UNet # Choose between UNet or Transformer     
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100 # Try more!



def test_unet():
    noise_steps = 1000
    model = backbone.UNET().to(device)
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
    image_size = 128
    num_classes = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model with 3 input channels
    model = backbone.DiT(input_size = image_size).to(device)
    model.eval()

    # Create dummy RGB input
    x = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,), dtype=torch.long).to(device)       # (1,)
    y = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 3, 32, 32)

    # Forward pass
    with torch.no_grad():
        out = model(x, t, y)

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {t.shape}")
    print(f"Label shape: {y.shape}")
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Output shape: {out.shape}\n")  # Expected: (1, out_channels, 32, 32)

def test_unet_with_dit():
    # Parameters
    batch_size = 1
    in_channels = 3       # e.g., 2 concatenated RGB images or multi-variate field
    image_size = 128
    noise_steps = 1000
   
    # Initialize model
    model = backbone.Flex(size = image_size).to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 6, 32, 32)
    t = torch.randint(0, noise_steps, (batch_size,), dtype=torch.long).to(device)  # (1,)
    c = torch.randn(1, 3, image_size, image_size).to(device)  # Example condition (flow parameters, e.g., 3 channels)

    # Forward pass
    with torch.no_grad():
        out = model(x, t , c)

    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {t.shape}")
    print(f"Condition shape: {c.shape}")
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Output shape: {out.shape}\n")  # Expected: (1, 6, 32, 32)


def test_UDiT():
    # Parameters
    batch_size = 1
    in_channels = 3   # RGB
    image_size = 128
    num_classes = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model with 3 input channels
    model = backbone.UDiT(input_size = image_size).to(device)
    model.eval()

    # Create dummy RGB input
    x = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,), dtype=torch.long).to(device)       # (1,)
    y = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 3, 32, 32)

    # Forward pass
    with torch.no_grad():
        out = model(x, t, y)
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {t.shape}")
    print(f"Label shape: {y.shape}")
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Output shape: {out.shape}\n")  # Expected: (1, out_channels, 32, 32)

def test_unet_with_uvit():
    # Parameters
    batch_size = 1
    in_channels = 3       # e.g., 2 concatenated RGB images or multi-variate field
    image_size = 128
    noise_steps = 1000
   
    # Initialize model
    model = backbone.UTFLEX(size = image_size).to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 6, 32, 32)
    t = torch.randint(0, noise_steps, (batch_size,), dtype=torch.long).to(device)  # (1,)
    c = torch.randn(1, 3, image_size, image_size).to(device)  # Example condition (flow parameters, e.g., 3 channels)

    # Forward pass
    with torch.no_grad():
        out = model(x, t , c)
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {t.shape}")
    print(f"Condition shape: {c.shape}")
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Output shape: {out.shape}\n")  # Expected: (1, 6, 32, 32)


def sample_diffusion(technique="ddpm", timestep=300,skip = 5, plot = False):
    # Parameters
    batch_size = 1
    in_channels = 3       # e.g., 2 concatenated RGB images or multi-variate field
    image_size = 128
    noise_steps = timestep
    inputs = torch.randn(batch_size, in_channels, image_size, image_size).to(device)  # (1, 3, 32, 32)
    # Initialize model
    model = backbone.UNetWithTransformer(noise_steps=noise_steps, time_dim=256, size =image_size).to(device).to(device)
    model.eval()

    diffuser = diff.Diffuser(timesteps=timestep, device=device, sample_trajectory_factor=skip)  # Adjust timesteps and device as needed
    prediction=diffuser.sample_from_noise(model,inputs,Tech = technique)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {prediction.shape}")  # Expected: (1, 3, 32, 32)
    if plot:
        prep.plot(prediction)
    return prediction




def test_dataloader(batch_size=5, img_size=64, visualize=True):
    print(f"Testing DataLoader with batch_size={batch_size}, img_size={img_size}")
    train_loader, val_loader, test_loader = unifoil.get_and_load_dataset(batch_size=batch_size, img_size=img_size)

    def inspect_loader(name, loader):
        try:
            for batch in loader:
                print(f"{name} batch shape: {batch.shape}")
                if visualize:
                    first_tensor = batch[0]  # Only the first tensor in the batch
                    print(f"Visualizing first tensor from {name} set...")
                    #plot(first_tensor)
                break  # Only inspect the first batch
        except Exception as e:
            print(f"❌ Error loading {name} batch: {e}")

    inspect_loader("Train", train_loader)
    inspect_loader("Validation", val_loader)
    inspect_loader("Test", test_loader)
    print("✅ DataLoader test complete.")

'''     
                Run Tests 
'''
#Test Models 
#test_unet()
#test_Transformer()
#test_unet_with_dit()
#test_UDiT()
#test_unet_with_uvit()

# Test data

#test_dataloader(batch_size =  1, img_size = 128, visualize = True)

# Test Diffuser 

# Trainer 


