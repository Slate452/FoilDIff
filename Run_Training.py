#!/home/kogbuagu/pytorch_env/bin/python

import os
import sys
import Unifoil_pipeline as unifoil
import Diffuser as diff
import Backbone
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.optim import Adam
import Trainer as T
import tests
import OpenFoam_pipeline as prep
import zipfile 




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('./models', exist_ok=True)
huge_loader, big_loader, small_loader = unifoil.get_and_load_dataset(batch_size=10, img_size=32)
image_size = prep.IMG_SIZE
batchsize = prep.BATCH_SIZE
noise_steps = 200

epochs = 1000 # Try more!
def save_path(model_name):
    return f'./models/{model_name}.pth'

#Models with presaved configs
CNNUnet = Backbone.UNET().to(device)
Unet = Backbone.UNET().to(device)
Tran = Backbone.DiT(input_size = image_size).to(device)
UNetTran = Backbone.Flex(size = image_size).to(device)
UViT = Backbone.UDiT(input_size = image_size).to(device)
UNetUViT = Backbone.UTFLEX(size = image_size).to(device)


combined_dataset, aux_train, aux_test, means, stds = prep.get_and_load_dataset()


def Diffusion_Train(save_path= save_path, predictor = None , loader = aux_test) :
    diffuser = diff.CosSchDiffuser(steps=noise_steps, device="cuda")
    # Adjust timesteps and device as needed
    trainer = T.Trainer(model=predictor, diffuser=diffuser, data_loader = loader, epochs= epochs, lr=1e-4, device=device, )
    # Start training
    Trained_model = trainer.train()
    path = str(save_path(predictor.__class__.__name__))

    Trained_model = Trained_model["last_model"]
    Optimizer =  trainer.optimizer.state_dict()
    # Save the model
    torch.save({
                    'model_state_dict': Trained_model.state_dict(),
                    'optimizer_state_dict': Optimizer,
                    }, path)
    print(f"Model saved to {path}")

    return Trained_model


UTFLex = Diffusion_Train(save_path, predictor = UNetUViT)