
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from customDataset import MaskDataset
import matplotlib.pyplot as plt



train_set = MaskDataset(csv_file='training.csv', root_dir='dataset_resized\\training', transform=transforms.ToTensor())
#test_set = MaskDataset(csv_file='testing.csv', root_dir='dataset_resized\\testing', transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_set, batch_size=len(train_set))

data = next(iter(train_loader))
print("mean:" + str(data[0].mean()))
print("std:" + str(data[0].std()))

#plt.hist(data[0].flatten())
#plt.avxline(data[0].mean())


#plt.show()

print("hi")




