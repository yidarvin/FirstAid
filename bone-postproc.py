import h5py
import numpy as np
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
import scipy.misc
import scipy.ndimage

path_imgs = "/media/dnr/8EB49A21B49A0C39/data/bone-orig/inference/inference"
path_save = "/media/dnr/8EB49A21B49A0C39/data/bone-orig/JonesBonesPart1C_PNG_pred"
if not isdir(path_save):
    mkdir(path_save)

def colormap(img, cmap):
    return_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    red_chan = np.clip(-2 + 4.0*cmap, 0,1)
    green_chan = np.clip(2 - 4.0*np.abs(cmap - 0.5), 0,1)
    blue_chan = np.clip(2 - 4.0*cmap, 0,1)
    return_img[:,:,0] = 0.2*red_chan + 0.8*img
    return_img[:,:,1] = 0.2*green_chan + 0.8*img
    return_img[:,:,2] = 0.2*blue_chan + 0.8*img
    return return_img

def gray2rgb(img):
    return_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    return_img[:,:,0] = img
    return_img[:,:,1] = img
    return_img[:,:,2] = img
    return return_img

list_imgs = listdir(path_imgs)
for name_img in list_imgs:
    if name_img[0] == '.':
        continue
    if name_img[-3:] != '.h5':
        continue
    path_img = join(path_imgs, name_img)
    h5f = h5py.File(path_img)
    height = np.asarray(h5f.get('height'))
    width = np.asarray(h5f.get('width'))
    img = np.asarray(h5f.get('data')).squeeze()
    pred = np.asarray(h5f.get('seg_pred')).squeeze()
    pred_bin = scipy.ndimage.filters.gaussian_filter(pred, 5)
    pred_bin = (pred_bin > 0.5) + 0
    pred_bin = gray2rgb(pred_bin)
    img_gray = gray2rgb(img)
    img_col  = colormap(img, scipy.ndimage.filters.gaussian_filter(pred, 5))
    pred     = gray2rgb(pred)
    
    pred = scipy.misc.imresize(pred, [height, width])
    pred_bin = scipy.misc.imresize(pred_bin, [height, width])
    img_gray = scipy.misc.imresize(img_gray, [height, width])
    img_col = scipy.misc.imresize(img_col, [height, width])
    
    img_save = np.concatenate((np.concatenate((img_gray, pred), axis=1),
                               np.concatenate((pred_bin, img_col), axis=1)), axis=0)
    scipy.misc.imsave(join(path_save, str(np.asarray(h5f.get('name')))), img_save)
    h5f.close()
