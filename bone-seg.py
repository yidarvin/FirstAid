import h5py
import numpy as np
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
import scipy.misc

img_size = 512
path_imgs = "/media/dnr/8EB49A21B49A0C39/data/bone-orig/OriImages"
path_masks = "/media/dnr/8EB49A21B49A0C39/data/bone-orig/OriMask"
path_inf = "/media/dnr/8EB49A21B49A0C39/data/bone-orig/JonesBonesPart1C_PNG"

path_save = "/home/dnr/Documents/data/bone-seg"
if not isdir(path_save):
    mkdir(path_save)
path_train = join(path_save, "training")
path_validation = join(path_save, "validation")
path_inference = join(path_save, "inference")
if not isdir(path_train):
    mkdir(path_train)
if not isdir(path_validation):
    mkdir(path_validation)
if not isdir(path_inference):
    mkdir(path_inference)

list_imgs = listdir(path_imgs)
for name_img in list_imgs:
    if name_img[0] == '.':
        continue
    if name_img[-4:] != '.png':
        continue
    name_save = name_img[10:]
    name_mask = "BJMaskOri" + name_img[10:]
    path_img = join(path_imgs, name_img)
    path_mask = join(path_masks, name_mask)
    if not isfile(path_mask):
        continue
    # Converting image and mask to necessary format.
    img = scipy.misc.imread(path_img)
    img = img.astype(np.float32)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    height, width = img.shape
    img = scipy.misc.imresize(img, [img_size, img_size])
    img = img.reshape([img_size, img_size, 1])
    img = img.astype(np.float32)
    img /= 255
    mask = scipy.misc.imread(path_mask)
    mask = mask.astype(np.float32)
    if len(mask.shape) == 3:
        mask = np.mean(mask, axis=2)
    mask = scipy.misc.imresize(mask, [img_size, img_size])
    mask = (mask > 0) + 0
    mask = mask.astype(np.int32)
    # Save image and mask to h5.
    folder_save = name_save[:-4]
    if np.random.choice(100) == 1:
        if not isdir(join(path_validation, folder_save)):
            mkdir(join(path_validation, folder_save))
        path_save_file = join(path_validation,folder_save, name_save[:-4]+'.h5')
    else:
        if not isdir(join(path_train, folder_save)):
            mkdir(join(path_train, folder_save))
        path_save_file = join(path_train, folder_save, name_save[:-4]+'.h5')
    h5f = h5py.File(path_save_file, 'w')
    h5f.create_dataset('data', data=img)
    h5f.create_dataset('seg', data=mask)
    h5f.create_dataset('name', data=name_img)
    h5f.create_dataset('height', data=height)
    h5f.create_dataset('width', data=width)
    h5f.create_dataset('depth', data=1)
    h5f.close()

list_imgs = listdir(path_inf)
for name_img in list_imgs:
    if name_img[0] == '.':
        continue
    if name_img[-4:] != '.png':
        continue
    path_img = join(path_inf, name_img)
    # Converting image to necessary format
    img = scipy.misc.imread(path_img)
    img = img.astype(np.float32)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    height, width = img.shape
    img = scipy.misc.imresize(img, [img_size, img_size])
    img = img.reshape([img_size, img_size, 1])
    img = img.astype(np.float32)
    img /= 255
    # Save image to h5.
    folder_save = name_save[:-4]
    path_save_file = join(path_inference, name_img[:-4]+'.h5')
    h5f = h5py.File(path_save_file, 'w')
    h5f.create_dataset('data', data=img)
    h5f.create_dataset('name', data=name_img)
    h5f.create_dataset('height', data=height)
    h5f.create_dataset('width', data=width)
    h5f.create_dataset('depth', data=1)
    h5f.close()
