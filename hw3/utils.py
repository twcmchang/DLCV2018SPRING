import os
import numpy as np
import scipy.misc


def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = scipy.misc.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown
    return mask

def read_images(filepath):
    file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
    file_list.sort()
    n_images = len(file_list)
    images = np.empty((n_images, 512, 512, 3))

    for i, file in enumerate(file_list):
        img = scipy.misc.imread(os.path.join(filepath, file))
        img = img.astype(int)
        images[i] = img
    return images

