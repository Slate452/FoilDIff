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
from sympy import symbols, latex

# === Settings ===
DATA_DIR ='../'
FLOW_TENSOR_DIR = os.path.join(DATA_DIR,"data", "FTturbsims_npy_b3")
FULL_FLOW_DIR = os.path.join(DATA_DIR,"data","FTturbsims_npy_full_tensor")
INPUT_CSV = os.path.join(DATA_DIR,"data", "Airfoils_Cases_Simulations.csv")
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

def check_for_nans(tensor, tensor_name="tensor"):
    """Check for NaN values in tensor and report statistics"""
    if np.isnan(tensor).any():
        nan_count = np.sum(np.isnan(tensor))
        total_elements = tensor.size
        print(f"WARNING: {tensor_name} contains {nan_count}/{total_elements} NaN values ({100*nan_count/total_elements:.2f}%)")
        return True
    return False

def safe_replace_nans(tensor, replacement_value=0.0):
    """Replace NaN values with a safe replacement value"""
    if np.isnan(tensor).any():
        tensor = np.nan_to_num(tensor, nan=replacement_value, posinf=replacement_value, neginf=replacement_value)
    return tensor

############################## Get Flow Tensor ####################################
def parse_row(row):
    try:
        airfoil = str(row[0].strip()).replace(",", "").replace("'", "").replace("[", "").replace("]", "").replace(" ", "")
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
            Re = float(row[4])/10000000
            
            # Convert to radians
            AoA_rad = math.radians(AoA_deg)
            Recos = Re * math.cos(AoA_rad)
            Resin = Re * math.sin(AoA_rad)
            
            # Load tensor and check for NaNs
            tensor = np.load(flow_tensor_path)
            check_for_nans(tensor, "loaded tensor")
            
            mask, ux, uy, cp = tensor[0], tensor[1], tensor[2], tensor[3]
            H, W = mask.shape
            
            # Check individual channels for NaNs
            check_for_nans(mask, "mask")
            check_for_nans(ux, "ux")
            check_for_nans(uy, "uy")
            check_for_nans(cp, "cp")
            
            # Clean NaNs from input channels
            mask = safe_replace_nans(mask, 0.0)
            ux = safe_replace_nans(ux, 0.0)
            uy = safe_replace_nans(uy, 0.0)
            cp = safe_replace_nans(cp, 0.0)
            
            # Create inverse mask (1 where mask is 0, 0 where mask is 1)
            # Assuming mask=1 represents the airfoil solid region
            inverse_mask = (mask == 0).astype(np.float32)
            
            # Apply inverse mask to Reynolds number components
            # Only apply Re*cos(AoA) and Re*sin(AoA) to fluid regions (where mask=0)
            recos_arr = np.full((H, W), Recos, dtype=np.float32) * inverse_mask
            resin_arr = np.full((H, W), Resin, dtype=np.float32) * inverse_mask
            
            # Stack all channels
            full_tensor = np.stack([recos_arr, resin_arr, mask, ux, uy, cp], axis=0)
            
            # Final NaN check and cleanup
            check_for_nans(full_tensor, "full_tensor before saving")
            full_tensor = safe_replace_nans(full_tensor, 0.0)
            
            # Verify no NaNs remain
            if check_for_nans(full_tensor, "final full_tensor"):
                print(f"ERROR: NaNs still present in final tensor for {airfoil}_{case}")
                return None
            
            # Save the cleaned tensor
            np.save(fout_path, full_tensor)
            #print(f"✅ Processed and saved: {fout_name}")

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

############################## Generate Training, Test and validation splits ####################################

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
        
        # Additional safety check for NaNs during loading
        if np.isnan(full_tensor).any():
            print(f"WARNING: NaNs detected in loaded tensor: {row['full_flow_tensor_path']}")
            full_tensor = safe_replace_nans(full_tensor, 0.0)
        
        full_tensor = torch.from_numpy(full_tensor).float()
        
        # Check for NaNs in PyTorch tensor
        if torch.isnan(full_tensor).any():
            print(f"WARNING: NaNs detected in PyTorch tensor: {row['full_flow_tensor_path']}")
            full_tensor = torch.nan_to_num(full_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.transform:
            full_tensor = self.transform(full_tensor)
            
        # Final check after transforms
        if torch.isnan(full_tensor).any():
            print(f"WARNING: NaNs detected after transforms: {row['full_flow_tensor_path']}")
            full_tensor = torch.nan_to_num(full_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
        return full_tensor

############################## Dataset Loader ####################################
def get_and_load_dataset(batch_size=None, img_size=None):
    train_df, test_df, val_df = load_or_generate_splits()
    if batch_size is None: batch_size = BATCH_SIZE
    if img_size is None: img_size = IMG_SIZE

    f_transforms = transforms.Compose([
        transforms.Lambda(lambda t: F.interpolate(t.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)),
        transforms.Lambda(lambda t: torch.flip(t, dims=[1])),  
        transforms.Lambda(lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0))
    ])

    train_dataset = CompletedNumpyDatasetFromFileList(train_df, transform=f_transforms)
    test_dataset = CompletedNumpyDatasetFromFileList(test_df, transform=f_transforms)
    val_dataset = CompletedNumpyDatasetFromFileList(val_df, transform=f_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"train: {len(train_dataset)}, test: {len(test_dataset)}, val: {len(val_dataset)}")
    return train_loader, val_loader, test_loader

############################## Validation Function ####################################
def validate_dataset_for_nans(loader, dataset_name="dataset"):
    """Validate that no NaN values exist in the dataset"""
    print(f"Validating {dataset_name} for NaN values...")
    nan_batches = 0
    total_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        if torch.isnan(batch).any():
            nan_batches += 1
            print(f"NaN detected in batch {batch_idx}")
        total_batches += 1
        
        # Check first few batches only for efficiency
        if batch_idx >= 10:
            break
    
    if nan_batches == 0:
        print(f"✅ {dataset_name} validation passed - no NaN values detected")
    else:
        print(f"❌ {dataset_name} validation failed - {nan_batches}/{total_batches} batches contain NaN values")

############################## Visualization ####################################
def plot_tensor_channels(tensor, cmap='viridis', use="dataset"):
    num_channels = tensor.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(15, 15))

    # Define channel name sets as Python lists
    dataset = [r"$\mathrm{Re} \cos(\mathrm{\alpha})$", r"$\mathrm{Re} \sin(\mathrm{\alpha})$", r"$\mathrm{\Omega}$", r"$C_p$", r"$U_x$", r"$U_y$"]
    params = [r"$\mathrm{Re} \cos(\mathrm{\alpha})$", r"$\mathrm{Re} \sin(\mathrm{\alpha})$", r"$\mathrm{\Omega}$"]
    uncertainty = [r"$\overline{C_p}$", r"$\overline{U_x}$", r"$\overline{U_y}$", r"$\sigma_{C_p}$", r"$\sigma_{U_x}$", r"$\sigma_{U_y}$"]
    field = [r"$C_p$", r"$U_x$", r"$U_y$"]
    
    # Select which set to use
    if use == "dataset":
        channel_names = dataset
    elif use == "params":
        channel_names = params
    elif use == "uncertainty":
        channel_names = uncertainty
    elif use == "field":
        channel_names = field
    else:
        raise ValueError(f"Unknown use value: {use}")

    # Ensure axes is iterable
    if num_channels == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        channel_data = tensor[i]
        channel_data = np.rot90(channel_data, k=1)  # Rotate 90° anticlockwise

        if np.isnan(channel_data).any():
            print(f"WARNING: NaNs detected in channel {i} ({channel_names[i] if i < len(channel_names) else f'Channel {i}'})")
            channel_data = np.nan_to_num(channel_data, nan=0.0)

        im = ax.imshow(channel_data, cmap=cmap)
        ax.set_title(channel_names[i] if i < len(channel_names) else f"Channel {i}")
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def plot(case: torch.Tensor = torch.randn(6, 32, 32),use = None):
    case = case.cpu()
    
    # Check for NaNs before plotting
    if torch.isnan(case).any():
        print("WARNING: NaNs detected in tensor before plotting")
        case = torch.nan_to_num(case, nan=0.0)
    
    if case.ndim == 4 and case.size(0) > 1:
        for s in range(min(BATCH_SIZE, case.size(0))):
            d = case[s].squeeze(0)
            plot_tensor_channels(d.numpy(), cmap='viridis',use=use)
    else:
        np_array = case[0].squeeze(0).numpy() if case.ndim == 4 else case.numpy()
        plot_tensor_channels(np_array, cmap='viridis')
