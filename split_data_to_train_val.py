import os
import random

data_path = "images/"

all_files = os.listdir(data_path)
random.shuffle(all_files)

train_files = all_files[0:int(0.6 * len(all_files))]
val_files = all_files[int(0.6 * len(all_files)) : int(0.8 * len(all_files))]
test_files = all_files[int(0.8 * len(all_files)):]

print("MOVING TRAIN IMAGES")
count = 0
for train_file in train_files:
    os.rename(data_path+train_file, "train/"+train_file)
    count += 1
    if count % 200 == 0:
        print("\t",count," out of ", len(train_files))

print("MOVING VAL IMAGES")
count = 0
for val_file in val_files:
    os.rename(data_path+val_file, "val/"+val_file)
    count += 1
    if count % 200 == 0:
        print("\t",count," out of ", len(val_files))

print("MOVING TEST IMAGES")
count = 0
for test_file in test_files:
    os.rename(data_path+test_file, "test/"+test_file)
    count += 1
    if count % 200 == 0:
        print("\t",count," out of ", len(test_files))