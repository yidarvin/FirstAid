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
from nets_attention_seg import *
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
    exec_statement = "self.pred, self.seg, self.att = "
    #self.pred, self.seg, self.att =
    exec_statement += opts.network
    #self.pred, self.seg, self.att = GoogLe
    exec_statement += "_Net(self.xTe, self.is_training, "
    #self.pred, self.seg, self.att = GoogLe_Net(self.xTe, self.is_training,
    exec_statement += str(opts.num_class)
    #self.pred, self.seg, self.att = GoogLe_Net(self.xTe, self.is_training, 2
    exec_statement += ", 1"
    #self.pred, self.seg, self.att = GoogLe_Net(self.xTe, self.is_training, 2, 1
    exec_statement += ", self.keep_prob)"
    #self.pred, self.seg, self.att = GoogLe_Net(self.xTe, self.is_training, 2, 1, self.keep_prob)
    return exec_statement

def create_exec_statement_train(opts):
    """
    Same as create_exec_statement_test but for multi
    gpu parsed training cycles.
    INPUTS:
    - opts: (object) command line arguments from argparser
    """
    exec_statement = "pred,seg,att = "
    #pred,seg,att =
    exec_statement += opts.network
    #pred,seg,att = GoogLe
    exec_statement += "_Net(multi_inputs[i], self.is_training, "
    #pred,seg,att = GoogLe_Net(multi_inputs[i], self.is_training,
    exec_statement += str(opts.num_class)
    #pred,seg,att = GoogLe_Net(multi_inputs[i], self.is_training, 2
    exec_statement += ", "
    #pred,seg,att = GoogLe_Net(multi_inputs[i], self.is_training, 2,
    exec_statement += str(opts.batch_size / opts.num_gpu)
    #pred,seg,att = GoogLe_Net(multi_inputs[i], self.is_training, 2, 12
    exec_statement += ", self.keep_prob)"
    #pred,seg,att = GoogLe_Net(multi_inputs[i], self.is_training, 2, 12, self.keep_prob)
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

class classifier:
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
        yTe_size = [1]
        sTe_size = [1, self.matrix_size, self.matrix_size]
        each_bs  = self.opts.batch_size
        xTr_size = [each_bs, self.matrix_size, self.matrix_size, self.num_channels]
        yTr_size = [each_bs]
        sTr_size = [each_bs, self.matrix_size, self.matrix_size]
        self.xTe = tf.placeholder(tf.float32, xTe_size)
        self.yTe = tf.placeholder(tf.int64, yTe_size)
        self.sTe = tf.placeholder(tf.int64, sTe_size)
        self.xTr = tf.placeholder(tf.float32, xTr_size)
        self.yTr = tf.placeholder(tf.int64, yTr_size)
        self.sTr = tf.placeholder(tf.int64, sTr_size)
        self.is_training = tf.placeholder_with_default(1, shape=())
        self.keep_prob = tf.placeholder(tf.float32)

        # Creating the Network for Testing
        exec_statement = create_exec_statement_test(opts)
        exec exec_statement
        self.L2_loss = get_L2_loss(self.opts.l2)
        self.L1_loss = get_L1_loss(self.opts.l1)
        self.ce_loss = get_ce_loss(self.pred, self.yTe)
        self.seg_loss = get_seg_loss(self.seg, self.sTe, self.opts.num_class)
        self.cost = self.ce_loss + self.L2_loss + self.L1_loss + self.seg_loss
        self.prob = tf.nn.softmax(self.pred)
        self.seg_prob = tf.nn.softmax(self.seg)
        self.acc = get_accuracy(self.pred, self.yTe)

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
        acc_multi = []
        multi_inputs = tf.split(self.xTr, self.opts.num_gpu, 0)
        multi_outputs = tf.split(self.yTr, self.opts.num_gpu, 0)
        multi_segs = tf.split(self.sTr, self.opts.num_gpu, 0)
        tf.get_variable_scope().reuse_variables()
        for i in xrange(self.opts.num_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('gpu%d' % i) as scope:
                    exec_statement = create_exec_statement_train(opts)
                    exec exec_statement
                    loss = get_ce_loss(pred, multi_outputs[i])
                    loss_seg = get_seg_loss(seg, multi_segs[i], self.opts.num_class)
                    loss_multi.append(loss)
                    cost = loss + self.L2_loss + self.L1_loss + loss_seg

                    if i == 0:
                        prob_seg = tf.nn.softmax(seg)
                        self.segmentation_example = prob_seg[0]

                    grads_and_vars = optimizer.compute_gradients(cost)
                    grads_multi.append(grads_and_vars)

                    accuracy = get_accuracy(pred, multi_outputs[i])
                    acc_multi.append(accuracy)
        grads = average_gradients(grads_multi)
        self.optimizer = optimizer.apply_gradients(grads, global_step=global_step)
        self.loss_multi = tf.add_n(loss_multi) / self.opts.num_gpu
        self.acc_multi = tf.add_n(acc_multi) / self.opts.num_gpu

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=None)

        self.tr_acc = []
        self.tr_loss = []
        self.val_acc = []
        self.val_loss = []

        if self.opts.bool_display:
            self.f = plt.figure()
            self.plot_accuracy = self.f.add_subplot(121)
            self.plot_loss = self.f.add_subplot(122)
            self.f2, self.plot_att = plt.subplots(5,5)
            self.f3 = plt.figure()
            self.image_orig = self.f3.add_subplot(131)
            self.seg_pred = self.f3.add_subplot(132)
            self.seg_truth = self.f3.add_subplot(133)

        self.dataXX = np.zeros(xTr_size, dtype=np.float32)
        self.dataYY = np.zeros(yTr_size, dtype=np.int64)
        self.dataSS = np.zeros(sTr_size, dtype=np.int64)

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
            self.f3.savefig(join(path_save, name + '.png'))
        plt.pause(0.05)
        return 0
    
    def average_accuracy(self, logits, truth):
        prediction = np.argmax(logits, axis=1)
        return np.mean(0.0 + (prediction == truth))
    
    def super_graph(self, save=False, name='0'):
        self.plot_accuracy.cla()
        self.plot_loss.cla()

        self.plot_accuracy.plot(self.tr_acc, 'b')
        if self.val_acc:
            self.plot_accuracy.plot(self.val_acc, 'r')
        self.plot_accuracy.set_ylim([0,1])
        self.plot_accuracy.set_xlabel('Epoch')
        self.plot_accuracy.set_ylabel('Accuracy')
        self.plot_accuracy.set_title('Accuracy')

        self.plot_loss.plot(self.tr_loss, 'b')
        if self.val_loss:
            self.plot_loss.plot(self.val_loss, 'r')
        ymax = 2 * np.log(self.opts.num_class)
        self.plot_loss.set_ylim([0, ymax])
        self.plot_loss.set_xlabel('Epoch')
        self.plot_loss.set_ylabel('-log(P(correct_class))')
        self.plot_loss.set_title('CrossEntropy Loss')
        
        if self.opts.path_visualization and save:
            path_save = join(self.opts.path_visualization, 'accuracy')
            if not isdir(path_save):
                mkdir(path_save)
            self.f1.savefig(join(path_save, name + '.png'))
        plt.pause(0.05)
        return 0

    def super_graph_att(self, img, att, save=False, name='0'):
        for i in range(len(self.plot_att)):
            for j in range(len(self.plot_att[0])):
                self.plot_att[i][j].cla()
                if i==0 and j==0:
                    self.plot_att[0][0].imshow(img)
                else:
                    self.plot_att[i][j].imshow(att[0,:,:,i*len(self.plot_att)+j-1])

        if self.opts.path_visualization and save:
            path_save = join(self.opts.path_visualization, 'attention')
            if not isdir(path_save):
                mkdir(path_save)
            self.f2.savefig(join(path_save, name + '.png'))
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
                    data_label = np.array(hf.get('label'))
                    data_seg = np.array(hf.get('seg'))
            except:
                print 'Failed: ' + join(self.opts.path_train, self.X_tr[ind], img_filename)
                continue
            data_iter, data_seg = data_augment(data_iter, data_seg)
            self.dataXX[iter_data,:,:,:] = data_iter
            self.dataYY[iter_data]   = data_label
            self.dataSS[iter_data,:,:] = data_seg
        feed = {self.xTr:self.dataXX, self.is_training:1, self.yTr:self.dataYY, self.sTr:self.dataSS, self.keep_prob:self.opts.keep_prob}
        _, loss_iter, acc_iter, seg_example = self.sess.run((self.optimizer, self.loss_multi, self.acc_multi, self.segmentation_example), feed_dict=feed)
        if self.opts.bool_display:
            self.super_graph_seg(self.dataXX[0,:,:,0], seg_example[:,:,1], self.dataSS[0,:,:])
        return loss_iter, acc_iter

    def inference_one_iter(self, path_file):
        """
        Does one forward pass and returns the segmentation.
        INPUTS:
        - self: (object)
        - path_file: (str) path of the file to inference.
        """
        dataXX = np.zeros((1, self.matrix_size, self.matrix_size, self.num_channels))
        while(True):
            try:
                with h5py.File(path_file) as hf:
                    dataXX[0,:,:,:] = np.array(hf.get('data'))
                    break
            except:
                time.sleep(0.001)
        feed = {self.xTe:dataXX, self.is_training:0, self.keep_prob:1.0}
        prob = self.sess.run((self.prob), feed_dict=feed)
        prob = prob[0]
        return prob

    def test_one_iter(self, path_file, name='0', counter=0):
        """
        Does one forward pass and returns the segmentation.
        INPUTS:
        - self: (object)
        - path_file: (str) path of the file to inference.
        """
        dataXX = np.zeros((1, self.matrix_size, self.matrix_size, self.num_channels))
        dataYY = np.zeros((1))
        while(True):
            try:
                with h5py.File(path_file) as hf:
                    dataXX[0,:,:,:] = np.array(hf.get('data'))
                    dataYY[0]   = np.array(hf.get('label'))
                    break
            except:
                time.sleep(0.001)
        feed = {self.xTe:dataXX, self.is_training:0, self.yTe:dataYY, self.keep_prob:1.0}
        loss, acc, att = self.sess.run((self.ce_loss, self.acc, self.att), feed_dict=feed)
        if (counter == 0) and self.opts.bool_display:
            self.super_graph_att(dataXX[0,:,:,0], att)
        return loss, acc

    def test_all(self, path_X):
        """
        Basically tests all the folders in path_X.
        INPUTS:
        - self: (object)
        - path_X: (str) file path to the data.
        """
        # Initializing variables.
        X_list = listdir(path_X)
        acc_te  = 0.0
        loss_te = 0.0
        # Doing the testing.
        for i,iter_data in enumerate(range(len(X_list))):
            # Reading in the data.
            path_data_iter = join(path_X, X_list[iter_data])
            files_data_iter = listdir(path_data_iter)
            for j, file_data in enumerate(files_data_iter):
                path_file = join(path_data_iter, file_data)
                loss_iter_iter, acc_iter_iter = self.test_one_iter(path_file, name=file_data, counter=(i or j)+0)
                loss_te += loss_iter_iter / len(files_data_iter) / len(X_list)
                acc_te += acc_iter_iter / len(files_data_iter) / len(X_list)
        return loss_te, acc_te
        
    
    def train_model(self):
        """
        Loads model and trains.
        """
        if not self.opts.path_train:
            return 0
        # Initializing
        start_time = time.time()
        loss_tr = 0.0
        acc_tr = 0.0
        if self.opts.bool_load:
            self.saver.restore(self.sess, self.opts.path_model)
        else:
            self.sess.run(self.init)
        # Training
        self.super_print("Let's start the training!")
        for iter in range(self.iter_count):
            loss_temp, acc_temp = self.train_one_iter(iter)
            loss_tr += loss_temp / self.print_every
            acc_tr += acc_temp / self.print_every
            if ((iter)%self.print_every) == 0 or iter == self.iter_count-1:
                if iter == 0:
                    loss_tr *= self.print_every
                    acc_tr *= self.print_every
                self.tr_loss.append(loss_tr)
                self.tr_acc.append(acc_tr)
                current_time = time.time()
                statement = "\t"
                statement += "Iter: " + str(iter) + " "
                statement += "Time: " + str((current_time - start_time) / 60) + " "
                statement += "Loss_tr: " + str(loss_tr)
                loss_tr = 0.0
                acc_tr = 0.0
                if self.opts.path_validation:
                    loss_val, acc_val = self.test_all(self.opts.path_validation)
                    self.val_loss.append(loss_val)
                    self.val_acc.append(acc_val)
                    statement += " Loss_val: " + str(loss_val)
                if self.opts.bool_display:
                    self.super_graph()
                self.super_print(statement)
        if self.opts.path_model:
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
                prob = self.inference_one_iter(path_file)
                h5f = h5py.File(path_file, 'a')
                h5f.create_dataset('label_pred', data=prob)
                h5f.close()
            
            
                
