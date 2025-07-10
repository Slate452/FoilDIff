
import os 
from os import listdir
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


BATCH_SIZE = 15
IMG_SIZE = 64

def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines


def get_data_files(img_dir ="./data/data",test_file = './data/Test.txt', train_file = './data/Train.txt', Trainingsplit = 0.7):
        #Check if training and test list of files exist 
        train_list , test_list =[], []
        img_list  = os.listdir(img_dir)
        if "Train.txt" in os.listdir('./data'): 
                """find better variable or question """
                test_list, train_list = read_file_to_list(test_file), read_file_to_list(train_file)
                return train_list, test_list 
        else:
                test_file, train_file = open(train_file, 'w'), open(test_file, 'w')
                for i, img in enumerate(img_list):
                        if i < int(len(img_list)*Trainingsplit):
                                train_file.write(f"{img}\n")
                                train_list.append(img)
                                
                        else:
                                test_file.write(f"{img}\n")
                                test_list.append(img)
                return train_list , test_list

        

def show_tensor_image(img):
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


def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines

def get_data_files(img_dir ="./data/1_parameter/results",test_file = './data/1_parameter/test_cases.txt', train_file = './data/1_parameter/train_cases.txt', Trainingsplit = 0.7):
        #Check if training and test list of files exist 
        train_list , test_list =[], []
        img_list  = os.listdir(img_dir)
        if "train_cases.txt'" in os.listdir('./data/1_parameter/'): 
                """find better variable or question """
                test_list, train_list = read_file_to_list(test_file), read_file_to_list(train_file)
                return train_list, test_list 
        else:
                test_file, train_file = open(train_file, 'w'), open(test_file, 'w')
                for i, img in enumerate(img_list):
                        if i < int(len(img_list)*Trainingsplit):
                                train_file.write(f"{img}\n")
                                train_list.append(img)
                                
                        else:
                                test_file.write(f"{img}\n")
                                test_list.append(img)
                return train_list , test_list


class NumpyDatasetFromFileList(Dataset):
    def __init__(self, file_list, file_dir, transform=None):
        self.file_list = file_list
        self.file_dir = file_dir
        self.transform = transform  # Store the transform

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


def get_and_load_dataset(img_dir = "./data/1_parameter/results"):
        # Define transformations
        f_transforms = transforms.Compose([
            transforms.Lambda(lambda t: F.interpolate(t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0)),  # Resize tensor
            transforms.Lambda(lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)),  # to [0, 1]
            transforms.Lambda(lambda t: t * 2 - 1),  # to [-1, 1]
            transforms.Lambda(lambda t: t.to(dtype=torch.float32))
        ])
        # Read training and testing lists
        train_file_list, test_file_list = get_data_files()
        # Create datasets
        train_dataset = NumpyDatasetFromFileList(train_file_list, file_dir=img_dir, transform=f_transforms)
        test_dataset = NumpyDatasetFromFileList(test_file_list, file_dir=img_dir, transform=f_transforms)
        train_loader= DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True, drop_last=True)
        combined_dataset = ConcatDataset([train_dataset, test_dataset])
        loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        print(f"train dataset: {len(train_dataset)}")
        return combined_dataset, train_loader, test_loader


def plot_tensor_channels(tensor, cmap='viridis'):
    num_channels = tensor.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(15, 15))
    '''
    The Shape(Omega), angle of attack and Reynolds number for a snapshot are encoded in the first 3 channel.
    The Angle of attack(Alpha) and  Reynolds numberb(Re) are 
    '''
    channel_names = ["Uf.cos_alpha","Uf.cos_alpha","Omega","Pressure", "Ux", "Uy"]
    
    if num_channels == 1:
        axes = [axes]  
    
    for i, ax in enumerate(axes):
        im = ax.imshow(tensor[i], cmap=cmap)
        ax.set_title(channel_names[i])
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    plt.show()



def plot(case:torch.tensor = torch.randn(6, 32, 32)):
    case = case.cpu()
    # Convert to NumPy array
    if case.ndim == 4 and case.size(0) > 1:
        print("The tensor is a batch of images with more than one element.")
        for s in range(0,BATCH_SIZE):
            d = case[s].squeeze(0)
            np_array = d.numpy()
            plot_tensor_channels(np_array, cmap='viridis')
    else:
        if case.ndim == 4:
            print("The tensor is a batch of images with one or fewer elements.")
            case = case[0].squeeze(0)
            np_array = case.numpy()
            plot_tensor_channels(np_array, cmap='viridis')
        else:
            np_array = case.numpy()
            plot_tensor_channels(np_array, cmap='viridis')
