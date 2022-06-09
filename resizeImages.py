from PIL import Image
import os

datasets = ["training", "testing"]
labels = ["no_mask", "cloth", "surgical", "n95"]

#create directories
for dataset in datasets:
    for label in labels:
        filepath_resized = "dataset_resized/" + dataset + "/" + label
        if not os.path.exists(filepath_resized):
            os.makedirs(filepath_resized)

#get average & median resolution
avg_width = 0
avg_height = 0
imgcount = 0 #for avg res

#for median res
widthList = []
heightList = []

for dataset in datasets:
    for label in labels:
        fileList = os.listdir("dataset/" + dataset + "/" + label)
        for file in fileList:
            filepath = "dataset/" + dataset + "/" + label + "/" + file
            filepath_resized = "dataset_resized/" + dataset + "/" + label + "/" + file
            image = Image.open(filepath)
            width, height = image.size
            #avg
            avg_width += width
            avg_height += height
            imgcount += 1
            #median
            widthList.append(width)
            heightList.append(height)

avg_width = int(avg_width/imgcount)
avg_height = int(avg_height/imgcount)

widthList.sort()
heightList.sort()

medianWidth = widthList[int(len(widthList)/2)]
medianHeight = heightList[int(len(heightList)/2)]
print("avg    res: " + str(avg_width) + "x" + str(avg_height))
print("median res: " + str(medianWidth) + "x" + str(medianHeight))

print("img count:" + str(imgcount))

print("resizing imgs...")
#resize all images
for dataset in datasets:
    for label in labels:
        fileList = os.listdir("dataset/" + dataset + "/" + label)
        for file in fileList:
            filepath = "dataset/" + dataset + "/" + label + "/" + file
            filepath_resized = "dataset_resized/" + dataset + "/" + label + "/" + file
            image = Image.open(filepath)
            try:
                new_image = image.resize((avg_width, avg_height))
                new_image.save(filepath_resized)
            except:
                #print(filepath_resized)
                #if image.mode in ("RGBA", "P"):
                    #print("########################")
                new_image = image.resize((avg_width, avg_height)).convert("RGB")
                new_image.save(filepath_resized)

            #exit()

print("done")
