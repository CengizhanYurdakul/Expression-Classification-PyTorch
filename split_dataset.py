from sklearn.model_selection import train_test_split
import pandas as pd
import os
import shutil

df = pd.DataFrame(columns=["Name", "Class"])
dict_ = {}

path = "./CK+48/"
train_path = "./Dataset/train/"
test_path = "./Dataset/test/"

dirs = os.listdir(path)
classes = dirs.copy()

for i in dirs:
    n_dirs = os.listdir(os.path.join(path + i))
    for j in n_dirs:
        df = df.append({"Name": j, "Class": i}, ignore_index=True)

print("DataFrame Created!")

x = df.iloc[:, 0]
y = df.iloc[:, 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

print("Splitted!")

for name_train, label_train in zip(x_train, y_train):
    if not os.path.exists(os.path.join(train_path + label_train)):
        os.makedirs(os.path.join(train_path + label_train))
    shutil.move(os.path.join(path + label_train, name_train), os.path.join(train_path + label_train, name_train))

for name_test, label_test in zip(x_test, y_test):
    if not os.path.exists(os.path.join(test_path + label_test)):
        os.makedirs(os.path.join(test_path + label_test))
    shutil.move(os.path.join(path + label_test, name_test), os.path.join(test_path + label_test, name_test))

for i in dirs:
    os.removedirs(path + i)

print("DONE!")