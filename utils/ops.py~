import numpy as np
import tensorflow as tf

from layers import *

def get_L2_loss(reg_param, key="reg_variables"):
    """
    L2 Loss Layer. Usually will use "reg_variables" collection.
    INPUTS:
    - reg_param: (float) the lambda value for regularization.
    - key: (string) the key for the tf collection to get from.
    """
    L2_loss = 0.0
    for W in tf.get_collection(key):
        L2_loss += reg_param * tf.nn.l2_loss(W)
    return L2_loss

def get_L1_loss(reg_param, key="reg_variables"):
    """
    L1 Loss Layer. Usually will use "reg_variables" collection.
    INPUTS:
    - reg_param: (float) the lambda value for regularization.
    - key: (string) the key for the tf collection to get from.
    """
    L1_loss = 0.0
    for W in tf.get_collection(key):
        L1_loss += reg_param * tf.reduce_sum(tf.abs(W))
    return L1_loss

def get_seg_loss(logits, labels, num_class):
    logits = tf.reshape(logits, [-1, num_class])
    labels = tf.reshape(labels, [-1])
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def get_optimizer(lr, decay, epoch_every):
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = float(lr)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               epoch_every, decay, staircase=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)#tf.train.AdamOptimizer(learning_rate)#
    return optimizer, global_step
