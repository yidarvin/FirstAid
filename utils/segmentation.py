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
from nets_segmentation import *
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
    exec_statement += str(opts.batch_size / max(1,opts.num_gpu))
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
        elif self.opts.path_inference:
            self.matrix_size, self.num_channels = find_data_shape(self.opts.path_inference)
        else:
            self.matrix_size, self.num_channels = 512,1
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
        multi_inputs = tf.split(self.xTr, max(1,self.opts.num_gpu), 0)
        multi_outputs = tf.split(self.yTr, max(1,self.opts.num_gpu), 0)
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
        if self.opts.num_gpu == 0:
            i = 0
            with tf.name_scope('cpu0') as scope:
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
        self.loss_multi = tf.add_n(loss_multi) / max(1,self.opts.num_gpu)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=None)

        if self.opts.bool_display:
            self.f1 = plt.figure()
            self.image_orig = self.f1.add_subplot(131)
            self.seg_pred = self.f1.add_subplot(132)
            self.seg_truth = self.f1.add_subplot(133)

        self.dataXX = np.zeros(xTr_size, dtype=np.float32)
        self.dataYY = np.zeros(yTr_size, dtype=np.int64)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def average_iou(self, pred, truth):
        img_pred = np.argmax(pred, axis=2)
        iou = 0.0
        for i in range(pred.shape[2]):
            intersection = np.sum((img_pred == i) & (truth == i))
            union = np.sum((img_pred == i) | (truth == i))
            iou += float(intersection) / float(union) / pred.shape[2]
        return iou
    
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
    
    def super_graph_seg(self, img, pred, truth, save=False, name='0'):
        self.image_orig.cla()
        self.seg_pred.cla()
        self.seg_truth.cla()
        self.image_orig.imshow(img, cmap='bone')
        self.seg_pred.imshow(self.super_colormap(img, pred))
        self.seg_truth.imshow(self.super_colormap(img, truth))
        self.image_orig.set_title('Original')
        self.seg_pred.set_title('Prediction')
        self.seg_truth.set_title('Ground Truth')
        if self.opts.path_visualization and save:
            path_save = join(self.opts.path_visualization, 'segmentation')
            if not isdir(path_save):
                mkdir(path_save)
            self.f1.savefig(join(path_save, name + '.png'))
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
            try:
                with h5py.File(join(self.opts.path_train, self.X_tr[ind], img_filename)) as hf:
                    data_iter = np.array(hf.get('data'))
                    data_seg = np.array(hf.get('seg'))
            except:
                print 'Failed: ' + join(self.opts.path_train, self.X_tr[ind], img_filename)
                continue
            data_iter, data_seg = data_augment(data_iter, data_seg)
            self.dataXX[iter_data,:,:,:] = data_iter
            self.dataYY[iter_data,:,:]   = data_seg
        feed = {self.xTr:self.dataXX, self.is_training:1, self.yTr:self.dataYY}
        _, loss_iter,seg_example = self.sess.run((self.optimizer, self.loss_multi, self.segmentation_example), feed_dict=feed)
        if self.opts.bool_display:
            self.super_graph_seg(self.dataXX[0,:,:,0], seg_example[:,:,1], self.dataYY[0,:,:])
        return loss_iter

    def inference_one_iter(self, path_file):
        """
        Does one forward pass and returns the segmentation.
        INPUTS:
        - self: (object)
        - path_file: (str) path of the file to inference.
        """
        dataXX = np.zeros((1, self.matrix_size, self.matrix_size, self.num_channels))
        try:
            with h5py.File(path_file) as hf:
                dataXX[0,:,:,:] = np.array(hf.get('data'))
        except:
            print 'Failed: ' + path_file
        feed = {self.xTe:dataXX, self.is_training:0}
        img = dataXX[0,:,:,0]
        mask = self.sess.run((self.prob), feed_dict=feed)
        mask = mask[0]
        mask = mask[:,:,1]
        rand = np.random.rand(mask.shape[0], mask.shape[1])
        if self.opts.bool_display:
            self.super_graph_seg(img, mask, rand)
        return mask

    def test_one_iter(self, path_file, name='0'):
        """
        Does one forward pass and returns the segmentation.
        INPUTS:
        - self: (object)
        - path_file: (str) path of the file to inference.
        """
        dataXX = np.zeros((1, self.matrix_size, self.matrix_size, self.num_channels))
        dataYY = np.zeros((1, self.matrix_size, self.matrix_size))
        try:
            with h5py.File(path_file) as hf:
                dataXX[0,:,:,:] = np.array(hf.get('data'))
                dataYY[0,:,:]   = np.array(hf.get('seg'))
        except:
            print 'Failed: ' + path_file
        feed = {self.xTe:dataXX, self.is_training:0, self.yTe:dataYY}
        seg_loss, pred = self.sess.run((self.seg_loss, self.pred), feed_dict=feed)
        iou = self.average_iou(pred[0], dataYY[0])
        if self.opts.bool_display:
            self.super_graph_seg(dataXX[0,:,:,0], pred[0,:,:,1], dataYY[0,:,:],
                                 save=True, name=name)
        return seg_loss, iou

    def test_all(self, path_X):
        """
        Basically tests all the folders in path_X.
        INPUTS:
        - self: (object)
        - path_X: (str) file path to the data.
        """
        # Initializing variables.
        X_list = listdir(path_X)
        iou_te  = 0.0
        loss_te = 0.0
        # Doing the testing.
        for iter_data in range(len(X_list)):
            # Reading in the data.
            path_data_iter = join(path_X, X_list[iter_data])
            files_data_iter = listdir(path_data_iter)
            for file_data in files_data_iter:
                path_file = join(path_data_iter, file_data)
                loss_iter_iter, iou_iter_iter = self.test_one_iter(path_file, name=file_data)
                loss_te += loss_iter_iter / len(files_data_iter) / len(X_list)
                iou_te += iou_iter_iter / len(files_data_iter) / len(X_list)
        return loss_te, iou_te
        
    
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
        self.super_print("Let's start the training!")
        loss_min = 1000000
        for iter in range(self.iter_count):
            loss_temp = self.train_one_iter(iter)
            loss_tr += loss_temp / self.print_every
            if ((iter)%self.print_every) == 0 or iter == self.iter_count-1:
                if iter == 0:
                    loss_tr *= self.print_every
                current_time = time.time()
                statement = "\t"
                statement += "Iter: " + str(iter) + " "
                statement += "Time: " + str((current_time - start_time) / 60) + " "
                statement += "Loss_tr: " + str(loss_tr)
                loss_tr = 0.0
                if self.opts.path_validation:
                    loss_val, iou_val = self.test_all(self.opts.path_validation)
                    statement += " Loss_val: " + str(loss_val)
                    statement += " IOU_val: " + str(iou_val)
                    if loss_val < loss_min:
                        loss_min = loss_val
                        self.saver.save(self.sess, self.opts.path_model)
                self.super_print(statement)
        if (not self.opts.path_validation) and self.opts.path_model:
            self.saver.save(self.sess, self.opts.path_model)
                

    def test_model(self):
        """
        Loads model and test.
        """
        if not self.opts.path_test:
            return 0
        # Initializing
        start_time = time.time()
        loss_te = 0.0
        self.saver.restore(self.sess, self.opts.path_model)

    def do_inference(self):
        """
        Loads model and does inference.
        """
        if not self.opts.path_inference:
            return 0
        # Initializing
        start_time = time.time()
        loss_te = 0.0
        self.saver.restore(self.sess, self.opts.path_model)
        for name_folder in listdir(self.opts.path_inference):
            path_imgs = join(self.opts.path_inference, name_folder)
            for name_img in listdir(path_imgs):
                if name_img[0] == '.':
                    continue
                if name_img[-3:] != '.h5':
                    continue
                path_file = join(path_imgs, name_img)
                mask = self.inference_one_iter(path_file)
                h5f = h5py.File(path_file, 'a')
                h5f.create_dataset('seg_'+self.opts.name, data=mask)
                h5f.close()
            
            
                
