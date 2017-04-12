import csv
import h5py
import matplotlib.animation as animation
import numpy as np
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
from pylab import *
import scipy.misc
import tensorflow as tf
import socket
import sys
import time

def find_data_shape(path_data):
    """
    Reads in one piece of data to find out number of channels.
    INPUT:
    path_data - (string) path of data
    """
    statement = ''
    dir_data = listdir(path_data)
    matrix_size = 0
    num_channels = 0
    # Trying to look into each patient folder.
    for folder_data in dir_data:
        path_patient = join(path_data, folder_data)
        if not isdir(path_patient):
            continue
        dir_file = listdir(path_patient)
        # Trying to look at each image file.
        for name_file in dir_file:
            if name_file[-3:] != '.h5':
                continue
            path_file = join(path_patient, name_file)
            try:
                with h5py.File(path_file) as hf:
                    img = np.array(hf.get('data'))
                    matrix_size = img.shape[0]
                    num_channels = img.shape[2]
            except:
                statement += path_file + ' is not valid.\n'
            if matrix_size != 0:
                break
        if matrix_size != 0:
            break
    if matrix_size == 0:
        statement += "Something went wrong in finding out img dimensions.\n"
    sys.stdout.write(statement)
    return matrix_size, num_channels

def calculate_iters(data_count, epoch, batch_size):
    """
    Uses training path, max_epoch, and batch_size to calculate
    the number of iterations to run, how long an epoch is in
    iterations, and how often to print.
    INPUT:
    data_count - (int) length of data
    epoch - (int) max number of epochs
    batch_size - (int) size of batch
    """
    iter_count = int(np.ceil(float(epoch) * data_count / batch_size))
    epoch_every = int(np.ceil(float(iter_count) / epoch))
    print_every = min([1000, epoch_every])
    print_every = max([10, print_every])
    return iter_count, epoch_every, print_every

def data_augment(data_iter, data_seg=None, rand_seed=None):
    """
    Stochastically augments the single piece of data.
    INPUT:
    - data_iter: (3d ND-array) the single piece of data
    - data_seg: (2d ND-array) the corresponding segmentation
    """
    matrix_size = data_iter.shape[0]
    # Setting Seed
    if rand_seed is not None:
        np.random.seed(rand_seed)
    # Creating Random variables
    roller = np.round(float(matrix_size/7))
    ox, oy = np.random.randint(-roller, roller+1, 2)
    do_flip = np.random.randn() > 0
    num_rot = np.random.choice(4)
    pow_rand = np.clip(0.05*np.random.randn(), -.2, .2) + 1.0
    add_rand = np.clip(np.random.randn() * 0.1, -.4, .4)
    # Rolling
    data_iter = np.roll(np.roll(data_iter, ox, 0), oy, 1)
    if np.any(data_seg):
        data_seg = np.roll(np.roll(data_seg, ox, 0), oy, 1)
    # Left-right Flipping
    if do_flip:
        data_iter = np.fliplr(data_iter)
        if np.any(data_seg):
            data_seg = np.fliplr(data_seg)
    # Random 90 Degree Rotation
    data_iter = np.rot90(data_iter, num_rot)
    if np.any(data_seg):
        data_seg = np.rot90(data_seg, num_rot)
    # Raising/Lowering to a power
    #data_iter = data_iter ** pow_rand
    # Random adding of shade.
    data_iter += add_rand
    if np.any(data_seg):
        return data_iter, data_seg
    return data_iter
