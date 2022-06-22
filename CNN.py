import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from customDataset import MaskDataset
import csv
from ctypes import resize
import torch.optim as optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
batch_size = 16
learning_rate = 0.001

# # dataset has PILImage images of range [0, 1].
# # We transform them to Tensors of normalized range [-1, 1]
# transform = transforms.Compose(
#      [transforms.ToTensor(),
#       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# # choosing the training and  test images dataset
#
# TRAIN_PATH = '../Users/jonat/Desktop/AI p1/dataset/train/'
# TEST_PATH = '../Users/jonat/Desktop/AI p1/dataset/test'
#
#
# train_dataset = torchvision.datasets.ImageNet(root=TRAIN_PATH, train=True,
#                                               download=True, transform=transform)
#
# test_dataset = torchvision.datasets.ImageNet(root=TEST_PATH, train=False,
#                                              download=True, transform=transform)
#
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
#                                           shuffle=False)
train_set = MaskDataset(csv_file='training.csv', root_dir='dataset_resized\\training', transform=transforms.ToTensor())
test_set = MaskDataset(csv_file='testing.csv', root_dir='dataset_resized\\testing', transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

print(train_set.__getitem__(999))
print(test_set.__getitem__(222))

classes = ('noMask', 'clothMask', 'SurgicalMask', 'n95Mask')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 because of the color channels rgb (input channel size) , 6 output , kernel is 5
        self.pool = nn.MaxPool2d(2, 2)  # kernal size 2 , stright 2
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 should be equal to the output of first conv , 16 output , kernel is 5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # input , output
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)  # -> n, 10
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')