import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class MaskDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, resize = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.labels = ["no_mask", "cloth", "surgical", "n95"]
    
    def __len__(self):
        return len(self.annotations)
    

    def __getitem__(self, index):
        #img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        img_path = os.path.join(self.root_dir + "\\" + self.labels[self.annotations.iloc[index,1]], self.annotations.iloc[index, 0])
        #print(img_path)
        #print(self.root_dir)
        #print(self.labels[self.annotations.iloc[index,1]])
        #exit()
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = self.transform(image)
             
        
        return(image, y_label)

#xd = MaskDataset(csv_file='training.csv', root_dir='dataset\\training', transform=None)   
#print(xd.__getitem__(500))