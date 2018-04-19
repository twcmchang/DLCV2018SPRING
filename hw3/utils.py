import os
import numpy as np
import imageio
import skimage.transform

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 256, 256))

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        mask = skimage.transform.resize(mask,output_shape=(256,256),order=0,preserve_range=True,clip=False)
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown
    return masks

def read_images(filepath):
    file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
    file_list.sort()
    n_images = len(file_list)
    images = np.empty((n_images, 256, 256, 3))

    for i, file in enumerate(file_list):
        img = imageio.imread(os.path.join(filepath, file))
        img = img.astype(int)
        img = skimage.transform.resize(img,output_shape=(256,256),order=0,preserve_range=True,clip=False)
        images[i] = img
    return images

def read_list(track):
    n_images = len(track)
    Xtrack = np.empty((n_images, 256, 256, 3))
    for i, file in enumerate(track):
        print(file)
        img = imageio.imread(file+"_sat.jpg")
        img = img.astype(int)
        img = skimage.transform.resize(img,output_shape=(256,256),order=0,preserve_range=True,clip=False)
        Xtrack[i] = img

    n_images = len(track)
    Ytrack = np.empty((n_images, 256, 256))
    for i, file in enumerate(track):
        mask = imageio.imread(file+"_mask.png")
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        mask = skimage.transform.resize(mask,output_shape=(256,256),order=0,preserve_range=True,clip=False)
        Ytrack[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        Ytrack[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        Ytrack[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        Ytrack[i, mask == 2] = 3  # (Green: 010) Forest land 
        Ytrack[i, mask == 1] = 4  # (Blue: 001) Water 
        Ytrack[i, mask == 7] = 5  # (White: 111) Barren land 
        Ytrack[i, mask == 0] = 6  # (Black: 000) Unknown
    return Xtrack, Ytrack