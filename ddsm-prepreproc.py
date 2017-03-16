import dicom
import numpy as np
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
import scipy.misc

path_imgs = "/media/dnr/8EB49A21B49A0C39/data/DDSM-orig/mammograms"
path_masks = "/media/dnr/8EB49A21B49A0C39/data/DDSM-orig/mammograms_dso"
path_save = "/media/dnr/8EB49A21B49A0C39/data/DDSM-orig/masks"

list_imgs = listdir(path_imgs)
list_masks = listdir(path_masks)
for name_img in list_imgs:
    name_pat = name_img[:15]
    mask = None
    for name_mask in list_masks:
        if name_mask[:15] != name_pat:
            continue
        file_path = join(path_masks, name_mask)
        ds = dicom.read_file(file_path)
        if np.any(mask):
            mask += ds.pixel_array
        else:
            mask = ds.pixel_array
        file_save = join(path_save, name_img)
        scipy.misc.imsave(file_save, mask)
        
