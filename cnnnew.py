# Load libraries

import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# checking for device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# Transforms
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5514, 0.4944, 0.4712], [0.2963, 0.2895, 0.2941]),  # 0-1 to [-1,1] , formula (x-mean)/std
])

# Dataloader

# Path for training and testing directory

train_path = 'training_dataset_resized/training'
test_path = 'testing_ff/testing'
#specific_test_path = 'testing_dataset_resized_random/testing'

#Load train and test dataset from folder
train_loader = torchvision.datasets.ImageFolder(train_path, transform=transformer)
test_loader = torchvision.datasets.ImageFolder(test_path, transform=transformer)

folds = KFold(n_splits=10, shuffle=True) #create our KFold cross validation
dataset_loader = torch.utils.data.ConcatDataset((train_loader,test_loader)) #Concatenate our test and training dataset into one dataset

# categories
root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)

class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1
        #(width -filter +2padding/stride) + 1
        # Input shape= (256,3,150,150)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (256,12,150,150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (256,12,150,150)
        self.relu1 = nn.ReLU()
        # Shape= (256,12,150,150)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size be factor 2
        # Shape= (256,12,75,75)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape= (256,20,75,75)
        self.relu2 = nn.ReLU()
        # Shape= (256,20,75,75)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (256,32,75,75)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (256,32,75,75)
        self.relu3 = nn.ReLU()
        # Shape= (256,32,75,75)

        self.fc = nn.Linear(in_features=75 * 75 * 32, out_features=num_classes)

        # Feed forwad function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        # Above output will be in matrix form, with shape (256,32,75,75)

        output = output.view(-1, 32 * 75 * 75)

        output = self.fc(output)

        return output

model = ConvNet(num_classes=6).to(device)

#Optmizer and loss function
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

num_epochs = 10

# calculating the size of training and testing images
train_count = len(glob.glob(train_path + '/**/*.jpg'))
test_count = len(glob.glob(test_path + '/**/*.jpg'))

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

print(train_count, test_count)
best_accuracy = 0.0
question = input("Do you want to load the model Y / N ")
loadModel = False
if question == "Y" or question == "y":
    loadModel = True
    model.load_state_dict((torch.load("best_checkpoint.model")))
    test_loader = DataLoader(test_loader,batch_size=1,shuffle=True)

if loadModel == True:
    CM = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs.data, 1)
            CM += confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1, 2, 3])

        tn = CM[0][0]
        tp = CM[1][1]
        fp = CM[0][1]
        fn = CM[1][0]
        accuracy = np.sum(np.diag(CM) / np.sum(CM))
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        print("Evaluation for test Set")
        print('\nTestset Accuracy(mean): %f %%' % (100 * accuracy))
        print()
        print('Confusion Matrix :')
        print(CM)
        print('- Recall : ', (tp / (tp + fn)) * 100, '%')
        print('- Precision: ', (tp / (tp + fp)) * 100, '%')
        print('- F1 : ', ((2 * recall * precision) / (recall + precision)) * 100, "%")

        print()

    # Show the CM
    df_cm = pd.DataFrame(CM, range(4), range(4))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 60}, xticklabels=['cloth', 'n95', 'no_mask', 'surgical'],
               yticklabels=['cloth', 'n95', 'no_mask', 'surgical'])  # font size

    plt.show()

elif loadModel == False:
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(dataset_loader)):
        print("---------------------- "+"fold number "+ str(i_fold+1)+" ----------------------")
        dataset_train = torch.utils.data.Subset(dataset_loader,train_idx)
        dataset_valid = torch.utils.data.Subset(dataset_loader, valid_idx)
        train_loader = DataLoader(dataset_train,batch_size=64)
        test_loader = DataLoader(dataset_valid,batch_size=1)

    # Model training and saving best model

        if loadModel == False:
            model.apply(reset_weights)
            train_accuracy_list = [None] * num_epochs
            for epoch in range(num_epochs):

                # Evaluation and training on training dataset
                model.train()
                train_accuracy = 0.0
                train_loss = 0.0

                for i, (images, labels) in enumerate(train_loader):
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                        labels = Variable(labels.cuda())

                    optimizer.zero_grad()

                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.cpu().data * images.size(0)
                    _, prediction = torch.max(outputs.data, 1)

                    train_accuracy += int(torch.sum(prediction == labels.data))

                train_accuracy = train_accuracy / train_count
                train_accuracy_list[epoch] = train_accuracy
                train_loss = train_loss / train_count
                print("Epoch ", epoch+1 ,"/", num_epochs, " : Train Loss:", train_loss, " Train Accuracy: ", train_accuracy)



        # CNN Network
            # Evaluation on testing dataset
        CM = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs.data, 1)
                CM += confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1, 2, 3])

            tn = CM[0][0]
            tp = CM[1][1]
            fp = CM[0][1]
            fn = CM[1][0]
            accuracy = np.sum(np.diag(CM) / np.sum(CM))
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            print("Evaluation for Fold #:" + str(i_fold +1))
            print('\nTestset Accuracy(mean): %f %%' % (100 * accuracy))
            print()
            print('Confusion Matrix :')
            print(CM)
            print('- Recall : ', (tp / (tp + fn)) * 100, '%')
            print('- Precision: ', (tp / (tp + fp)) * 100, '%')
            print('- F1 : ', ((2 * recall * precision) / (recall + precision)) * 100, "%")

            print()

    plt.plot(train_accuracy_list, '-o')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train'])
    plt.title('Train Accuracy')

    plt.show()

    #Show the CM
    df_cm = pd.DataFrame(CM, range(4), range(4))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 80},xticklabels=['cloth', 'n95','no_mask','surgical'],
                yticklabels=['cloth', 'n95','no_mask','surgical']) # font size

    plt.show()


    # Save the best model
if accuracy > best_accuracy:
    torch.save(model.state_dict(), 'best_checkpoint.model')
    best_accuracy = accuracy