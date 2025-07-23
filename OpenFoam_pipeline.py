import os 
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 10
IMG_SIZE = 32

def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines


def get_data_files(img_dir="./data/1_parameter/results", 
                   test_file='./data/1_parameter/test_cases.txt', 
                   train_file='./data/1_parameter/train_cases.txt', 
                   training_split=0.7):
    """Get training and test file lists, creating them if they don't exist"""
    train_list, test_list = [], []
    img_list = os.listdir(img_dir)
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        test_list, train_list = read_file_to_list(test_file), read_file_to_list(train_file)
        return train_list, test_list 
    else:
        with open(train_file, 'w') as train_f, open(test_file, 'w') as test_f:
            for i, img in enumerate(img_list):
                if i < int(len(img_list) * training_split):
                    train_f.write(f"{img}\n")
                    train_list.append(img)
                else:
                    test_f.write(f"{img}\n")
                    test_list.append(img)
        return train_list, test_list


def show_tensor_image(img):
    """Display tensor as image"""
    r_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(img.shape) == 4:
        img = img[0, :, :, :] 
    plt.imshow(r_transforms(img))


class NumpyDatasetFromFileList(Dataset):
    def __init__(self, file_list, file_dir, transform=None):
        self.file_list = file_list
        self.file_dir = file_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.file_dir, self.file_list[idx])
        with np.load(file_path) as data:
            np_array = data['a']  # Extract the array 'a' from the .npz file

        tensor = torch.from_numpy(np_array).float()  # Convert to PyTorch tensor

        # Apply transformation if defined
        if self.transform:
            tensor = self.transform(tensor)

        return tensor


def get_dataset_statistics(img_dir="./data/1_parameter/results"):
    """Calculate channel-wise mean and standard deviation across the entire dataset"""
    
    train_file_list, test_file_list = get_data_files(img_dir=img_dir)
    
    # Create datasets without any normalization transforms
    train_dataset = NumpyDatasetFromFileList(train_file_list, file_dir=img_dir)
    test_dataset = NumpyDatasetFromFileList(test_file_list, file_dir=img_dir)
    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    
   
    num_channels = 6
    sum_values = torch.zeros(num_channels)
    sum_squared_values = torch.zeros(num_channels)
    
    
    print(f"Calculating statistics for {len(combined_dataset)} samples...")
    
    for i in range(len(combined_dataset)):
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(combined_dataset)}")
            
        sample = combined_dataset[i]
        
        # Resize sample to target size for consistent statistics
        sample_resized = F.interpolate(sample.unsqueeze(0), size=(128, 128), 
                                     mode='bilinear', align_corners=False).squeeze(0)
        
        # Calculate statistics per channel
        for c in range(num_channels):
            channel_data = sample_resized[c]
            sum_values[c] += channel_data.sum()
            sum_squared_values[c] += (channel_data ** 2).sum()
        
        num_pixels = sample_resized.shape[1] * sample_resized.shape[2]
    
    # Calculate mean and std
    num_pixels_per_channel = num_pixels
    means = sum_values / num_pixels_per_channel
    variances = (sum_squared_values / num_pixels_per_channel) - (means ** 2)
    stds = torch.sqrt(variances + 1e-9)  
    
    print(f"Dataset statistics calculated:")
    print(f"Means: {means}")
    print(f"Stds: {stds}")
    
    return means, stds


def create_normalization_transform(means, stds):
    """Create normalization transform using calculated statistics"""
    def normalize_channels(tensor):
        """Normalize each channel using its mean and std"""
        for c in range(tensor.shape[0]):
            if c<2:
                tensor[c] = (tensor[c] - means[c]) / stds[c]
            elif c==2:
                tensor[c] = tensor[c]
            elif c>2:
                tensor[c] = (tensor[c] - means[c]) / stds[c]
            
        return tensor
    
    return transforms.Compose([
        transforms.Lambda(lambda t: F.interpolate(t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), 
                                                mode='bilinear', align_corners=False).squeeze(0)),
        transforms.Lambda(normalize_channels),
        transforms.Lambda(lambda t: t.to(dtype=torch.float32))
    ])


def get_and_load_dataset(img_dir="./data/1_parameter/results", use_cached_stats=True):
    """Load dataset with proper normalization using dataset statistics"""
    
    # Calculate or load dataset statistics
    if use_cached_stats:
        # Try to load cached statistics
        stats_file = os.path.join(img_dir, 'dataset_stats.pt')
        if os.path.exists(stats_file):
            print("Loading cached dataset statistics...")
            stats = torch.load(stats_file)
            means, stds = stats['means'], stats['stds']
        else:
            print("Calculating dataset statistics...")
            means, stds = get_dataset_statistics(img_dir)
            # Cache the statistics
            torch.save({'means': means, 'stds': stds}, stats_file)
            print(f"Statistics saved to {stats_file}")
    else:
        print("Calculating dataset statistics...")
        means, stds = get_dataset_statistics(img_dir)
    
    # Create normalization transform using calculated statistics
    f_transforms = create_normalization_transform(means, stds)
    
    # Read training and testing lists
    train_file_list, test_file_list = get_data_files(img_dir=img_dir)
    
    # Create datasets with normalization
    train_dataset = NumpyDatasetFromFileList(train_file_list, file_dir=img_dir, transform=f_transforms)
    test_dataset = NumpyDatasetFromFileList(test_file_list, file_dir=img_dir, transform=f_transforms)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    print(f"Train dataset: {len(train_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")
    print(f"Combined dataset: {len(combined_dataset)}")
    
    return combined_dataset, train_loader, test_loader, means, stds


def plot_tensor_channels(tensor, cmap='viridis'):
    """Plot individual channels of a tensor"""
    num_channels = tensor.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(15, 15))
    
    channel_names = ["Uf.cos_alpha", "Uf.sin_alpha", "Omega", "Pressure", "Ux", "Uy"]
    
    if num_channels == 1:
        axes = [axes]  
    
    for i, ax in enumerate(axes):
        im = ax.imshow(tensor[i], cmap=cmap)
        ax.set_title(channel_names[i] if i < len(channel_names) else f"Channel {i}")
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    plt.show()


def plot(case: torch.tensor = torch.randn(6, 32, 32)):
    """Plot tensor channels"""
    case = case.cpu()
    
    if case.ndim == 4 and case.size(0) > 1:
        print("The tensor is a batch of images with more than one element.")
        for s in range(min(BATCH_SIZE, case.size(0))):
            d = case[s]
            np_array = d.numpy()
            plot_tensor_channels(np_array, cmap='viridis')
    else:
        if case.ndim == 4:
            print("The tensor is a batch of images with one or fewer elements.")
            case = case[0]
            np_array = case.numpy()
            plot_tensor_channels(np_array, cmap='viridis')
        else:
            np_array = case.numpy()
            plot_tensor_channels(np_array, cmap='viridis')


#test
if __name__ == "__main__":
    # Load dataset with proper normalization
    combined_dataset, train_loader, test_loader, means, stds = get_and_load_dataset()
    
   