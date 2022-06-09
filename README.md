# Comp472-project
# AI Face Mask Detector
The goal is to develop an AI that can analyze facial images and detect whether a person is wearing a face mask or not, as well as what type of mask they are wearing.

# Dataset
This folder contains our picture dataset for testing and training. Inside these folders two folders, the pictures are sorted in the following categories: no_mask, cloth, surgical, n95.

# Dataset Resized
This folder contains our pre-processed dataset with a consistent resolution.

# training.csv, testing.csv
These files were used as dictionaries of file names and their labels. The first column is the filename, and the second column is an index from 0->3, which is mapped to the labels array ["no_mask", "cloth", "surgical", "n95"].

# ResizeImages.py
This script was used to pre-process our dataset. Initially, the mask pictures did not have a consistent resolution. To solve this problem, we found the average width and the average height across all images, resized them, and saved them into the dataset_resized folder. Later on, we realized that it was too large (512x519), so we divided that by 8. The final resolution is 64x64.

# CustomDataset.py
This script reads an input .csv file, and formats the data so that it can be used in pytorch's DataLoader() function.

# buildDatasetCSV.py
This script goes through the directories within the dataset. Then, it builds a .csv file, where each row contains: 
  1. the filename, 
  2. an index for their label.

# driver.py
A script to test our CustomDataset class + pytorch's DataLoader() function.

# How to Install
To run the main program cnnnew.py, you need the following libraries: pandas, PyTorch, scikit-image, and torchvision and matplotlib. 
If you want to test resizeImages.py and buildDatasetCSV.py, you will need these libraries: python image library (Pillow), CSV.

# How to Run
Simply Execute python cnnnew.py
