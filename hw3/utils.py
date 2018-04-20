import os
import numpy as np
import imageio
import skimage.transform


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

def read_masks(filepath, onehot=False):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    if onehot:
        masks = np.empty((n_masks, 256, 256, 7))
    else:
        masks = np.empty((n_masks, 256, 256))

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = skimage.transform.resize(mask,output_shape=(256,256),order=0,preserve_range=True,clip=False)
        if onehot:
            label_mask = rgb2label(mask)
            masks[i,:] = label2onehot(label_mask)
        else:
            masks[i,:] = rgb2label(mask)
    return masks

def rgb2label(rgb_mask):
    rgb_mask = (rgb_mask >= 128).astype(int)
    rgb_mask = 4 * rgb_mask[:, :, 0] + 2 * rgb_mask[:, :, 1] + rgb_mask[:, :, 2]
    label_mask = np.empty(rgb_mask.shape)
    label_mask[rgb_mask == 3] = 0  # (Cyan: 011) Urban land 
    label_mask[rgb_mask == 6] = 1  # (Yellow: 110) Agriculture land 
    label_mask[rgb_mask == 5] = 2  # (Purple: 101) Rangeland 
    label_mask[rgb_mask == 2] = 3  # (Green: 010) Forest land 
    label_mask[rgb_mask == 1] = 4  # (Blue: 001) Water 
    label_mask[rgb_mask == 7] = 5  # (White: 111) Barren land 
    label_mask[rgb_mask == 0] = 6  # (Black: 000) Unknown
    return label_mask

def label2onehot(label_mask):
    onehot_mask = np.empty((label_mask.shape[0], label_mask.shape[1], 7))
    onehot_mask[label_mask == 0,:] = np.array([1,0,0,0,0,0,0]) # 0 # (Cyan: 011) Urban land 
    onehot_mask[label_mask == 1,:] = np.array([0,1,0,0,0,0,0]) # 1  # (Yellow: 110) Agriculture land 
    onehot_mask[label_mask == 2,:] = np.array([0,0,1,0,0,0,0]) # 2  # (Purple: 101) Rangeland 
    onehot_mask[label_mask == 3,:] = np.array([0,0,0,1,0,0,0]) # 3  # (Green: 010) Forest land 
    onehot_mask[label_mask == 4,:] = np.array([0,0,0,0,1,0,0]) # 4  # (Blue: 001) Water 
    onehot_mask[label_mask == 5,:] = np.array([0,0,0,0,0,1,0]) # 5  # (White: 111) Barren land 
    onehot_mask[label_mask == 6,:] = np.array([0,0,0,0,0,0,1]) # 6  # (Black: 000) Unknown
    return onehot_mask

def onehot2label(onehot_mask):
    return np.argmax(onehot_mask, axis=2)

def label2rgb(label_mask):
    rgb_mask = np.empty(label_mask.shape)
    rgb_mask[label_mask == 0] = 3
    rgb_mask[label_mask == 1] = 6
    rgb_mask[label_mask == 2] = 5
    rgb_mask[label_mask == 3] = 2
    rgb_mask[label_mask == 4] = 1 
    rgb_mask[label_mask == 5] = 7
    rgb_mask[label_mask == 6] = 0
    final_rgb_mask = (np.dstack((rgb_mask//4,rgb_mask%4//2,rgb_mask%4%2))*255)
    return np.array(final_rgb_mask,dtype=np.uint8)

def read_list(track):
    n_images = len(track)
    Xtrack = np.empty((n_images, 256, 256, 3))
    for i, file in enumerate(track):
        img = imageio.imread(file+"_sat.jpg")
        img = img.astype(int)
        img = skimage.transform.resize(img,output_shape=(256,256),order=0,preserve_range=True,clip=False)
        Xtrack[i] = img

    n_images = len(track)
    Ytrack = np.empty((n_images, 256, 256, 7))
    for i, file in enumerate(track):
        mask = imageio.imread(file+"_mask.png")
        mask = skimage.transform.resize(mask,output_shape=(256,256),order=0,preserve_range=True,clip=False)
        label_mask = rgb2label(mask)
        Ytrack[i,:] = label2onehot(label_mask)
    return Xtrack, Ytrack