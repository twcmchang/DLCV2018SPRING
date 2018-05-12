import os
import numpy as np
import pandas as pd
import imageio
import skimage.transform

def read_dataset(csvfile, directory):
    df = pd.read_csv(csvfile)
    df.image_name = [os.path.join(directory, name) for name in df.image_name.values]
    images = []
    for filename in df.image_name.values:
        img = imageio.imread(filename)
        images.append(img/255.0)
    return np.array(images, dtype=np.float32), df

# def read_train_dataset():
#     train = pd.read_csv("hw4_data/train.csv")
#     train.image_name = [os.path.join("hw4_data/train/", name) for name in train.image_name.values]
#     images = []
#     for filename in train.image_name.values:
#         img = imageio.imread(filename)
#         images.append(img)
#     return np.array(images, dtype=np.float32), train
    
# def read_test_dataset():
#     test = pd.read_csv("hw4_data/test.csv")
#     test.image_name = [os.path.join("hw4_data/test/", name) for name in test.image_name.values]
#     images = []
#     for filename in test.image_name.values:
#         img = imageio.imread(filename)
#         images.append(img)
#     return np.array(images, dtype=np.float32), test
