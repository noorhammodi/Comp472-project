import os
import csv

labels = ["no_mask", "cloth", "surgical", "n95"]

def renameAllImgs():
    #utility, not meant to be run more than 1x bc not checking if file alrdy exists xd
    for index, label in enumerate(labels):
        fileList = os.listdir("dataset/training/" + label)
        for fileindex, file in enumerate(fileList):
            filepath = os.path.dirname(__file__) + "/dataset/training/" + label + "/"
            oldName = filepath+file
            targetName = filepath + "[" + label + "]_" + str(fileindex) + os.path.splitext(file)[1]
            os.rename(oldName, targetName)


def buildCSV(type="training"):
    if (type != "training" and type != "testing"):
        print("wr0ng type, valid types: 'training', 'testing'")
        return
    print("writing " + type + " csv")
    with open(type + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        #fileList = os.listdir("dataset_resized/testing")
        for index, label in enumerate(labels):
            fileList = os.listdir("dataset_resized/" + type + "/" + label)
            for file in fileList:
                writer.writerow([file, index])
        f.close()

#renameAllImgs()
buildCSV("training")
buildCSV("testing")
print("done")
