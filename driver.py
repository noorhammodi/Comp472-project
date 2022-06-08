import csv
from ctypes import resize
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from customDataset import MaskDataset

train_set = MaskDataset(csv_file='training.csv', root_dir='dataset_resized\\training', transform=transforms.ToTensor())
test_set = MaskDataset(csv_file='testing.csv', root_dir='dataset_resized\\testing', transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

print(train_set.__getitem__(999))
print(test_set.__getitem__(222))