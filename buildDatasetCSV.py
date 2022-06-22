import os
import csv
import uuid


labels = ["no_mask", "cloth", "surgical", "n95"]
datasets = ["training", "age/adult", "age/underage", "gender/male", "gender/female"]

def randomizeImageNames():
    for dataset in datasets:
        basePath = "dataset/training/"
        if dataset != "training":
            basePath = "dataset/testing/" + dataset + "/"
        for index, label in enumerate(labels):
            fileList = os.listdir(basePath + label)
            for fileindex, file in enumerate(fileList):
                filepath = os.path.dirname(__file__) + "/" + basePath + label + "/"
                oldName = filepath+file
                targetName = filepath + str(uuid.uuid4()) + str(uuid.uuid4()) + str(fileindex) + os.path.splitext(file)[1]
                os.rename(oldName, targetName)

def renameAllImgs():
    for dataset in datasets:
        basePath = "dataset/training/"
        extraLabel = ""
        if dataset != "training":
            basePath = "dataset/testing/" + dataset + "/"
            extraLabel = dataset.replace("/", "-") + "-"
        for index, label in enumerate(labels):
            fileList = os.listdir(basePath + label)
            for fileindex, file in enumerate(fileList):
                filepath = os.path.dirname(__file__) + "/" + basePath + label + "/"
                oldName = filepath+file

                targetName = filepath + "[" + extraLabel + label + "]_" + str(fileindex) + os.path.splitext(file)[1]
                os.rename(oldName, targetName)

def buildCSVs():
    for dataset in datasets:
        basePath = "training_dataset_resized/training/"
        csvFileName = "training.csv"
        if dataset != "training":
            break
            basePath = "dataset_resized/testing/" + dataset + "/"
            csvFileName = "testing_" + dataset.replace("/", "-") + ".csv"
        with open(csvFileName, 'w', newline='') as f:
            writer = csv.writer(f)
            #fileList = os.listdir("dataset_resized/testing")
            for index, label in enumerate(labels):
                fileList = os.listdir(basePath + "/" + label)
                for file in fileList:
                    writer.writerow([file, index])
            f.close()

#randomizeImageNames()
#renameAllImgs()
buildCSVs()

print("done")
