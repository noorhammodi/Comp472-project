# Comp472-project
# AI Face Mask Detector
The goal is to develop an AI that can analyze face images and detect whether a person is wearing a face mask or not, as well as what type of mask they are wearing.

# Dataset
This folder contains our picture dataset for testing and training. Inside these folders two folders, the pictures are sorted in the following categories : no_mask, cloth, surgicalm, n95.

# Dataset Resized
This folder contains our pre-processed dataset with a consistent resolution.

# training.csv, training.csv
These files were used as dictionaries of file names and their labels. The first column is the filename, and the second column is an index from 0->3, which is mapped to the labels array ["no_mask", "cloth", "surgical", "n95"].


# ResizeImages.py
This script was used to pre-process our dataset. Initially, the mask pictures did not have a consistent resolution. To solve this problem, we found the average width and the average height across all images, resized them, and saved them into the dataset_resized folder.

# CustomDataset.py
......

# buildDataset.py
......

# driver.py
......

# How to Install
......

# How to Run
......
