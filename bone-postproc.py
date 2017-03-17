import h5py
import numpy as np
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
import scipy.misc

path_imgs = ""
path_save = "/media/dnr/8EB49A21B49A0C39/data/bone-orig/JonesBonesPart2_PNG_pred"
if not isdir(path_save):
    mkdir(path_save)

list_imgs = listdir(path_imgs)
for name_img in list_imgs:
    if name_img[0] == '.':
        continue
    if name_img[-3:] == '.h5':
        continue
    path_img = join(path_imgs, name_img)
    h5f = h5f.File(path_img)
    height = np.asarray(h5f.get('height'))
    width = np.asarray(h5f.get('width'))
    img = np.asarray(h5f.get('data')).squeeze()
    img = scipy.misc.imresize(img, [height, width])
    pred = np.asarray(h5f.get('seg_pred')).squeeze()
    pred_scale = pred - np.min(pred)
    pred_scale /= np.max(pred_scale)
    pred = scipy.misc.imresize(pred, [height, width])
    pred_scale = scipy.misc.imresize(pred_scale, [height, width])
    img_save = np.concatenate((img, pred, pred_scale), axis=1)
    scipy.misc.imsave(join(path_save, h5f.get('name')), img_save)
