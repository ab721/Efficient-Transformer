#Learned how to write custom datasets from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import os
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

class MyDataset(Dataset):
    def __init__(self, data_directory, augmentation, image_names):
        self.data_directory = data_directory
        self.augmentation = augmentation
        self.image_names = image_names

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = io.imread(f'{self.data_directory}/Images/{self.image_names[index]}.jpg')
        mask = io.imread(f'{self.data_directory}/Labels/{self.image_names[index]}.png', as_gray = True)
        mask = np.expand_dims(mask, axis = -1)
        augmented_data = self.augmentation(image = image, mask = mask)
        image = augmented_data['image']
        mask = augmented_data['mask']
        image = torch.from_numpy(image.transpose((2,0,1)))
        mask = torch.from_numpy(mask.transpose((2,0,1)))
        return (image, mask)

def get_dataloader(data_directory, augmentation, image_names, shuffle, batch_size, num_workers):

    dataset = MyDataset(data_directory, augmentation, image_names)

    if shuffle:
        shuffle = True
    else:
        shuffle = False
        
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

    return dataloader