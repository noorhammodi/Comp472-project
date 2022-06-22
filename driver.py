
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from customDataset import MaskDataset
import matplotlib.pyplot as plt
import torch

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

train_set = MaskDataset(csv_file='training.csv', root_dir='dataset_resized\\training', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_set, batch_size=64)

data = next(iter(train_loader))
#print("mean:" + str(data[0].mean()))
#print("std:" + str(data[0].std()))
mean, std = get_mean_and_std(train_loader)
print("mean:" + str(mean))
print("std:" + str(std))


#plt.hist(data[0].flatten())
#plt.avxline(data[0].mean())


#plt.show()

print("hi")




