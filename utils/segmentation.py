import matplotlib.animation as animation
import h5py
import numpy as np
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
import scipy
import scipy.misc
import scipy.ndimage
from sklearn.metrics import roc_curve, auc,roc_auc_score
import tensorflow as tf
import socket
import sys
import time

from layers import *
from nets import *

def create_exec_statement_test(opts):
    """
    Creates an executable statement string.
    Basically lets us keep everything general.
    Comments show an example.
    INPUTS:
    - opts: (object) command line arguments from argparser
    """
    exec_statement = "self.pred = "
    #self.pred =
    exec_statement += opts.network
    #self.pred = GoogLe
    exec_statement += "_Net(self.x, self.is_training, "
    #self.pred = GoogLe_Net(self.x, self.is_training,
    exec_statement += str(opts.num_class)
    #self.pred = GoogLe_Net(self.x, self.is_training, 2
    exec_statement += ")"
    #self.pred = GoogLe_Net(self.x, self.is_training, 2)
    return exec_statement

def create_exec_statement_train(opts):
    """
    Same as create_exec_statement_test but for multi
    gpu parsed training cycles.
    INPUTS:
    - opts: (object) command line arguments from argparser
    """
    exec_statement = "pred = "
    #pred =
    exec_statement += opts.network
    #pred = GoogLe
    exec_statement += "_Net(multi_inputs[i], self.is_training, "
    #pred = GoogLe_Net(multi_inputs[i], self.is_training,
    exec_statement += str(opts.num_class)
    #pred = GoogLe_Net(multi_inputs[i], self.is_training, 2
    exec_statement += ")"
    #pred = GoogLe_Net(multi_inputs[i], self.is_training, 2)
    return exec_statement

def average_gradients(grads_multi):
    """
    Basically averages the aggregated gradients.
    Much was stolen from code from the Tensorflow team.
    Basically, look at the famous inceptionv3 code.
    INPUTS:
    - grads_multi: a list of gradients and variables
    """
    average_grads = []
    for grad_and_vars in zip(*grads_multi):
        grads = []
        for g,_ in grad_and_vars:
            if g is None:
                continue
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        if grads == []:
            continue
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class segmentor:
    def __init__(self, opts):
        """
        Initialization of all the fields.
        We also create the network.
        INPUTS:
        - opts: (object) command line arguments from argparser
        """
        self.opts = opts
        
