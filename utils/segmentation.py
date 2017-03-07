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
from data import *
from ops import *

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
    exec_statement += "_Net(self.xTe, self.is_training, "
    #self.pred = GoogLe_Net(self.xTe, self.is_training,
    exec_statement += str(opts.num_class)
    #self.pred = GoogLe_Net(self.xTe, self.is_training, 2
    exec_statement += ", 1)"
    #self.pred = GoogLe_Net(self.xTe, self.is_training, 2, 1)
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
    exec_statement += ", "
    #pred = GoogLe_Net(multi_inputs[i], self.is_training, 2,
    exec_statement += str(opts.batch_size / opts.num_gpu)
    #pred = GoogLe_Net(multi_inputs[i], self.is_training, 2, 12
    exec_statement += ")"
    #pred = GoogLe_Net(multi_inputs[i], self.is_training, 2, 12)
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
        grad = tf.concat(grads, 0)
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

        # Creating the Placeholders.
        if self.opts.path_train:
            self.matrix_size, self.num_channels = find_data_shape(self.opts.path_train)
        elif self.opts.path_test:
            self.matrix_size, self.num_channels = find_data_shape(self.opts.path_test)
        else:
            self.matrix_size, self.num_channels = 224,1
        xTe_size = [1, self.matrix_size, self.matrix_size, self.num_channels]
        yTe_size = [1, self.matrix_size, self.matrix_size]
        each_bs  = self.opts.batch_size
        xTr_size = [each_bs, self.matrix_size, self.matrix_size, self.num_channels]
        yTr_size = [each_bs, self.matrix_size, self.matrix_size]
        self.xTe = tf.placeholder(tf.float32, xTe_size)
        self.yTe = tf.placeholder(tf.int64, yTe_size)
        self.xTr = tf.placeholder(tf.float32, xTr_size)
        self.yTr = tf.placeholder(tf.int64, yTr_size)
        self.is_training = tf.placeholder_with_default(1, shape=())

        # Creating the Network for Testing
        exec_statement = create_exec_statement_test(opts)
        exec exec_statement
        self.L2_loss = get_L2_loss(self.opts.l2)
        self.L1_loss = get_L1_loss(self.opts.l1)
        self.seg_loss = get_seg_loss(self.pred, self.yTe, self.opts.num_class)
        self.cost = self.seg_loss + self.L2_loss + self.L1_loss
        self.prob = tf.nn.softmax(self.pred)

        # Listing the data.
        if self.opts.path_train:
            self.X_tr = listdir(self.opts.path_train)
            self.iter_count, self.epoch_every, self.print_every = calculate_iters(len(self.X_tr), self.opts.max_epoch, self.opts.batch_size)
        else:
            self.iter_count, self.epoch_every, self.print_every = calculate_iters(1000, self.opts.max_epoch, self.opts.batch_size)
        if self.opts.path_validation:
            self.X_val = listdir(self.opts.path_validation)
        if self.opts.path_test:
            self.X_te = listdir(self.opts.path_test)
        optimizer,global_step = get_optimizer(self.opts.lr, self.opts.lr_decay, self.epoch_every)
        grads = optimizer.compute_gradients(self.cost)
        self.optimizer = optimizer.apply_gradients(grads, global_step=global_step)

        # Creating the Network for Training
        loss_multi = []
        grads_multi = []
        multi_inputs = tf.split(self.xTr, self.opts.num_gpu, 0)
        multi_outputs = tf.split(self.yTr, self.opts.num_gpu, 0)
        tf.get_variable_scope().reuse_variables()
        for i in xrange(self.opts.num_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('gpu%d' % i) as scope:
                    exec_statement = create_exec_statement_train(opts)
                    exec exec_statement
                    loss = get_seg_loss(pred, multi_outputs[i], self.opts.num_class)
                    loss_multi.append(loss)
                    cost = loss + self.L2_loss + self.L1_loss

                    if i == 0:
                        prob = tf.nn.softmax(pred)
                        self.segmentation_example = prob[0]

                    grads_and_vars = optimizer.compute_gradients(cost)
                    grads_multi.append(grads_and_vars)
        grads = average_gradients(grads_multi)
        self.optimizer = optimizer.apply_gradients(grads, global_step=global_step)
        self.loss_multi = tf.add_n(loss_multi) / self.opts.num_gpu

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=None)

        if self.opts.bool_display:
            self.f = plt.figure()
            self.image_orig = self.f.add_subplot(131)
            self.seg_pred = self.f.add_subplot(132)
            self.seg_truth = self.f.add_subplot(133)

        self.dataXX = np.zeros(xTr_size, dtype=np.float32)
        self.dataYY = np.zeros(yTr_size, dtype=np.int64)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def super_colormap(self, img, cmap):
        img -= np.min(img)
        img /= np.max(img)
        return_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
        red_chan = np.clip(-2 + 4.0*cmap, 0,1)
        green_chan = np.clip(2 - 4.0*np.abs(cmap - 0.5), 0,1)
        blue_chan = np.clip(2 - 4.0*cmap, 0,1)
        return_img[:,:,0] = 0.2*red_chan + 0.8*img
        return_img[:,:,1] = 0.2*green_chan + 0.8*img
        return_img[:,:,2] = 0.2*blue_chan + 0.8*img
        return return_img
    
    def super_graph_seg(self, img, pred, truth, i=0):
        self.image_orig.cla()
        self.seg_pred.cla()
        self.seg_truth.cla()
        self.image_orig.imshow(img, cmap='bone')
        self.seg_pred.imshow(self.super_colormap(img, pred))
        self.seg_truth.imshow(self.super_colormap(img, truth))
        #if self.bool_movie:
        #    path_save = join(self.path_movie, 'segmentation')
        #    if not isdir(path_save):
        #        mkdir(path_save)
        #    self.f2.savefig(join(path_save, str(i) + '.png'))
        plt.pause(0.05)
        return 0

    def update_init(self):
        self.init = tf.global_variables_initializer()

    def super_print(self, statement):
        """
        This basically prints everything in statement.
        We'll print to stdout and path_log.
        """
        sys.stdout.write(statement + '\n')
        sys.stdout.flush()
        f = open(self.opts.path_log, 'a')
        f.write(statement + '\n')
        f.close()
        return 0

    def train_one_iter(self, i):
        """
        Basically trains one iteration.
        INPUTS:
        - self: (object)
        - i: (int) iteration
        """
        # Filling in the data.
        ind_list = np.random.choice(range(len(self.X_tr)), self.opts.batch_size, replace=True)
        for iter_data, ind in enumerate(ind_list):
            img_filename = np.random.choice(listdir(join(self.opts.path_train, self.X_tr[ind])))
            while(True):
                print join(self.opts.path_train, self.X_tr[ind], img_filename)
                try:
                    with h5py.File(join(self.opts.path_train, self.X_tr[ind], img_filename)) as hf:
                        data_iter = np.array(hf.get('data'))
                        data_seg = np.array(hf.get('seg'))
                    break
                except:
                    time.sleep(0.001)
            data_iter, data_seg = data_augment(data_iter, data_seg)
            self.dataXX[iter_data,:,:,:] = data_iter
            self.dataYY[iter_data,:,:]   = data_seg
        feed = {self.xTr:self.dataXX, self.is_training:1, self.yTr:self.dataYY}
        _, loss_iter,seg_example = self.sess.run((self.optimizer, self.loss_multi, self.segmentation_example), feed_dict=feed)
        self.super_graph_seg(self.dataXX[0,:,:,0], seg_example[:,:,1], self.dataYY[0,:,:], i)
        return loss_iter

    def train_model(self):
        """
        Loads model and trains.
        """
        if not self.opts.path_train:
            return 0
        # Initializing
        start_time = time.time()
        loss_tr = 0.0
        if self.opts.bool_load:
            self.saver.restore(self.sess, self.opts.path_model)
        else:
            self.sess.run(self.init)
        # Training
        for iter in range(self.iter_count):
            loss_temp = self.train_one_iter(iter)
            loss_tr += loss_temp / self.print_every
