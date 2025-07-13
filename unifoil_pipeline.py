import os 
import csv
import ast
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv
from glob import glob

# === Settings ===
DATA_DIR ='/mnt/d/Unifoil/'
FLOW_TENSOR_DIR = os.path.join(DATA_DIR, "FTturbsims_npy_b3")
FULL_FLOW_DIR = os.path.join(DATA_DIR, "FTturbsims_npy_full_tensor")
INPUT_CSV = os.path.join(DATA_DIR, "Run", "Airfoils_Cases_Simulations.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "FoilDiff","data", "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "FoilDiff", "data", "test.csv")
VAL_CSV = os.path.join(DATA_DIR, "FoilDiff", "data", "validate.csv")
OUTPUT_COLUMNS = ["airfoil", "case", "mach", "AoA", "Re", "cd", "cl", "convergence_file", "flow_tensor_path", "full_flow_tensor_path"]
FOLDERS = ["Turb_Cutout_1", "Turb_Cutout_2"]
CROP_BOUNDS = (-1.0, 2, -1.5, 1.5, -1e-3, 1e-3)
W, H = 128, 128
TARGET_BLOCK_INDEX = 2
BATCH_SIZE = 15
IMG_SIZE = H

############################## Helpers ####################################
def get_flow_tensor_path(filename):
    return os.path.join(FLOW_TENSOR_DIR, filename)

def get_full_flow_tensor_path(filename):
    return os.path.join(FULL_FLOW_DIR, filename)

def pad_tensor_to_size(tensor, target_size=(H, W)):
    c, h, w = tensor.shape
    target_h, target_w = target_size
    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Tensor shape {h}x{w} is larger than target size {target_h}x{target_w}")

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)


############################## Get Flow Tensor ####################################
def parse_row(row):
    try:
        airfoil =str(row[0].strip()).replace(",", "").replace("'", "").replace("[", "").replace ("]", "").replace(" ", "")
        case = row[1]
        mach = float(row[2])
        flow_tensor_name = f"{airfoil}_{case}_000_Mach_{mach:.4f}_flow_tensor.npy"
        flow_tensor_path = os.path.join(FLOW_TENSOR_DIR, flow_tensor_name)

    

        if not os.path.exists(flow_tensor_path):
            return None
            
        fout_name = f"{airfoil}_{case}_Mach_{mach:.4f}_full_flow_tensor.npy"
        fout_path = os.path.join(FULL_FLOW_DIR, fout_name)

        if not os.path.exists(fout_path): 
            AoA_deg = float(row[3])
            Re = float(row[4])
            AoA_rad = math.radians(AoA_deg)
            Recos = Re * math.cos(AoA_rad)
            Resin = Re * math.sin(AoA_rad)
            tensor = np.load(flow_tensor_path)
            mask, ux, uy, cp = tensor[0], tensor[1], tensor[2], tensor[3]
            H, W = mask.shape
            roi_mask = (mask == 1).astype(np.float32)
            recos_arr = np.full((H, W), Recos, dtype=np.float32) * roi_mask
            resin_arr = np.full((H, W), Resin, dtype=np.float32) * roi_mask
            full_tensor = np.stack([recos_arr, resin_arr, mask, ux, uy, cp], axis=0)
            np.save(fout_path, full_tensor)

        return {
                "airfoil": airfoil,
                "case": case,
                "mach": mach,
                "AoA": float(row[3]),
                "Re": float(row[4]),
                "cd": float(row[5]),
                "cl": float(row[6]),
                "convergence_file": row[7],
                "flow_tensor_path": flow_tensor_name,
                "full_flow_tensor_path": fout_name
            }
    except Exception as e:
        print(f"Failed to parse row: {row} | Error: {e}")
    return None

############################## Generate Training, Test and  validation splits ####################################

def load_or_generate_splits():
    if all(os.path.exists(p) for p in [TRAIN_CSV, TEST_CSV, VAL_CSV]):
        print("Loading existing splits...")
        return pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV), pd.read_csv(VAL_CSV)

    all_data = []
    with open(INPUT_CSV, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            parsed = parse_row(row)
            if parsed:
                all_data.append(parsed)

    random.shuffle(all_data)
    total = len(all_data)
    train_df = pd.DataFrame(all_data[:int(0.7*total)], columns=OUTPUT_COLUMNS)
    test_df = pd.DataFrame(all_data[int(0.7*total):int(0.9*total)], columns=OUTPUT_COLUMNS)
    val_df = pd.DataFrame(all_data[int(0.9*total):], columns=OUTPUT_COLUMNS)

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    print("✅ Data splits created and saved.")
    return train_df, test_df, val_df

############################## Dataset ####################################
class CompletedNumpyDatasetFromFileList(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        fout_path = get_full_flow_tensor_path(row['full_flow_tensor_path'])
        full_tensor = np.load(fout_path)
        full_tensor = torch.from_numpy(full_tensor).float()
        if self.transform:
            full_tensor = self.transform(full_tensor)
        return full_tensor

############################## Dataset Loader ####################################
def get_and_load_dataset(batch_size =None,img_size= None):
    train_df, test_df, val_df = load_or_generate_splits()
    if batch_size is None: batch_size = BATCH_SIZE
    if img_size is None: img_size = IMG_SIZE

    f_transforms = transforms.Compose([
                                        transforms.Lambda(lambda t: F.interpolate(t.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False ).squeeze(0)),
                                        transforms.Lambda(lambda t: torch.flip(t, dims=[1])) # Flip along height (dim=1)       
    ])

    train_dataset = CompletedNumpyDatasetFromFileList(train_df, transform=f_transforms)
    test_dataset = CompletedNumpyDatasetFromFileList(test_df, transform=f_transforms)
    val_dataset = CompletedNumpyDatasetFromFileList(val_df, transform=f_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    
    print(f"train: {len(train_dataset)}, test: {len(test_dataset)}, val: {len(val_dataset)}")
    return  train_loader, val_loader, test_loader

############################## Visualization ####################################
def plot_tensor_channels(tensor, cmap='viridis'):
    num_channels = tensor.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(15, 15))
    channel_names = ["Re*cos(AoA)", "Re*sin(AoA)", "Mask", "Ux", "Uy", "Cp"]

    if num_channels == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        im = ax.imshow(tensor[i], cmap=cmap)
        ax.set_title(channel_names[i] if i < len(channel_names) else f"Channel {i}")
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def plot(case: torch.Tensor = torch.randn(6, 32, 32)):
    case = case.cpu()
    if case.ndim == 4 and case.size(0) > 1:
        for s in range(min(BATCH_SIZE, case.size(0))):
            d = case[s].squeeze(0)
            plot_tensor_channels(d.numpy(), cmap='viridis')
    else:
        np_array = case[0].squeeze(0).numpy() if case.ndim == 4 else case.numpy()
        plot_tensor_channels(np_array, cmap='viridis')



