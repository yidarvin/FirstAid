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
from nets_classification import *
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
    exec_statement += ", 1"
    #self.pred = GoogLe_Net(self.xTe, self.is_training, 2, 1
    exec_statement += ", self.keep_prob)"
    #self.pred = GoogLe_Net(self.xTe, self.is_training, 2, 1, self.keep_prob)
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
    exec_statement += ", self.keep_prob)"
    #self.pred = GoogLe_Net(self.xTe, self.is_training, 2, 12, self.keep_prob)
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
        each_bs  = self.opts.batch_size
        xTr_size = [each_bs, self.matrix_size, self.matrix_size, self.num_channels]
        yTr_size = [each_bs]
        self.xTe = tf.placeholder(tf.float32, xTe_size)
        self.yTe = tf.placeholder(tf.int64, yTe_size)
        self.xTr = tf.placeholder(tf.float32, xTr_size)
        self.yTr = tf.placeholder(tf.int64, yTr_size)
        self.is_training = tf.placeholder_with_default(1, shape=())
        self.keep_prob = tf.placeholder(tf.float32)

        # Creating the Network for Testing
        exec_statement = create_exec_statement_test(opts)
        exec exec_statement
        self.L2_loss = get_L2_loss(self.opts.l2)
        self.L1_loss = get_L1_loss(self.opts.l1)
        self.ce_loss = get_ce_loss(self.pred, self.yTe)
        self.cost = self.ce_loss + self.L2_loss + self.L1_loss
        self.prob = tf.nn.softmax(self.pred)
        self.acc = get_accuracy(self.pred, self.yTe)

        # Listing the data.
        if self.opts.path_train:
            list_imgs = listdir(self.opts.path_train)
            for name_img in list_imgs:
                if name_img[0]=='.':
                    list_imgs.remove(name_img)
            self.X_tr = list_imgs
            self.iter_count, self.epoch_every, self.print_every = calculate_iters(len(self.X_tr), self.opts.max_epoch, self.opts.batch_size)
        else:
            self.iter_count, self.epoch_every, self.print_every = calculate_iters(1000, self.opts.max_epoch, self.opts.batch_size)
        if self.opts.path_validation:
            list_imgs = listdir(self.opts.path_validation)
            for name_img in list_imgs:
                if name_img[0] == '.':
                    list_imgs.remove(name_img)
            self.X_val = list_imgs
        if self.opts.path_test:
            list_imgs = listdir(self.opts.path_test)
            for name_img in list_imgs:
                if name_img[0] == '.':
                    list_imgs.remove(name_img)
            self.X_te = list_imgs
        optimizer,global_step = get_optimizer(self.opts.lr, self.opts.lr_decay, self.epoch_every)
        grads = optimizer.compute_gradients(self.cost)
        self.optimizer = optimizer.apply_gradients(grads, global_step=global_step)

        # Creating the Network for Training
        loss_multi = []
        grads_multi = []
        acc_multi = []
        multi_inputs = tf.split(self.xTr, max(self.opts.num_gpu,1), 0)
        multi_outputs = tf.split(self.yTr, max(self.opts.num_gpu,1), 0)
        tf.get_variable_scope().reuse_variables()
        for i in xrange(self.opts.num_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('gpu%d' % i) as scope:
                    exec_statement = create_exec_statement_train(opts)
                    exec exec_statement
                    loss = get_ce_loss(pred, multi_outputs[i])
                    loss_multi.append(loss)
                    cost = loss + self.L2_loss + self.L1_loss

                    grads_and_vars = optimizer.compute_gradients(cost)
                    grads_multi.append(grads_and_vars)

                    accuracy = get_accuracy(pred, multi_outputs[i])
                    acc_multi.append(accuracy)
        if self.opts.num_gpu == 0:
            i = 0
            with tf.name_scope('cpu0') as scope:
                exec_statement = create_exec_statement_train(opts)
                exec exec_statement
                loss = get_ce_loss(pred, multi_outputs[i])
                loss_multi.append(loss)
                cost = loss + self.L2_loss + self.L1_loss

                grads_and_vars = optimizer.compute_gradients(cost)
                grads_multi.append(grads_and_vars)

                accuracy = get_accuracy(pred, multi_outputs[i])
                acc_multi.append(accuracy)
        grads = average_gradients(grads_multi)
        self.optimizer = optimizer.apply_gradients(grads, global_step=global_step)
        self.loss_multi = tf.add_n(loss_multi) / max(self.opts.num_gpu,1)
        self.acc_multi = tf.add_n(acc_multi) / max(self.opts.num_gpu,1)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=None)

        self.tr_acc = []
        self.tr_loss = []
        self.val_acc = []
        self.val_loss = []

        if self.opts.bool_display:
            self.f1 = plt.figure()
            self.plot_accuracy = self.f1.add_subplot(121)
            self.plot_loss = self.f1.add_subplot(122)

        self.dataXX = np.zeros(xTr_size, dtype=np.float32)
        self.dataYY = np.zeros(yTr_size, dtype=np.int64)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def average_accuracy(self, logits, truth):
        prediction = np.argmax(logits, axis=1)
        return np.mean(0.0 + (prediction == truth))
    
    def confusion_matrix(self, logits, truth):
        prediction = np.argmax(logits, axis=1)
        truth = truth.astype(np.int64)
        prediction = prediction.astype(np.int64)
        O = np.zeros((self.opts.num_class, self.opts.num_class))
        for i in range(len(truth)):
            O[truth[i], prediction[i]] += 1
        return O
    
    def quadratic_kappa(self, logits, truth):
        prediction = np.argmax(logits, axis=1)
        truth = truth.astype(np.int64)
        prediction = prediction.astype(np.int64)
        t_vec = np.zeros((self.opts.num_class))
        p_vec = np.zeros((self.opts.num_class))
        O = np.zeros((self.opts.num_class, self.opts.num_class))
        for i in range(len(truth)):
            O[truth[i], prediction[i]] += 1
            t_vec[truth[i]] += 1
            p_vec[prediction[i]] += 1
        W = np.zeros((self.opts.num_class, self.opts.num_class))
        for i in range(self.opts.num_class):
            for j in range(self.opts.num_class):
                W[i,j] = ((i - j)**2) / ((self.opts.num_class - 1)**2)
        E = np.outer(t_vec, p_vec)
        O = O.astype(np.float32)
        W = W.astype(np.float32)
        E = np.sum(O) * E / np.sum(E)
        kappa = 1 - np.sum(W * O) / np.sum(W * E)
        return kappa
    
    def super_graph(self, save=True, name='0'):
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
                try:
                    with h5py.File(join(self.opts.path_train, self.X_tr[ind], img_filename)) as hf:
                        data_iter = np.array(hf.get('data'))
                        data_label = np.array(hf.get('label'))
                    break
                except:
                    time.sleep(0.001)
            data_iter = data_augment(data_iter)
            self.dataXX[iter_data,:,:,:] = data_iter
            self.dataYY[iter_data]   = data_label
        feed = {self.xTr:self.dataXX, self.is_training:1, self.yTr:self.dataYY, self.keep_prob:self.opts.keep_prob}
        _, loss_iter, acc_iter = self.sess.run((self.optimizer, self.loss_multi, self.acc_multi), feed_dict=feed)
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

    def test_one_iter(self, path_file, name='0'):
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
        loss, acc, pred = self.sess.run((self.ce_loss, self.acc, self.pred), feed_dict=feed)
        return loss, acc, pred, dataYY

    def test_all(self, path_X):
        """
        Basically tests all the folders in path_X.
        INPUTS:
        - self: (object)
        - path_X: (str) file path to the data.
        """
        # Initializing variables.
        X_list = listdir(path_X)
        for name in X_list:
            if name[0] == '.':
                X_list.remove(name)
        acc_te  = 0.0
        loss_te = 0.0
        preds = []
        truths = []
        counter = 0
        # Doing the testing.
        for iter_data in range(len(X_list)):
            # Reading in the data.
            path_data_iter = join(path_X, X_list[iter_data])
            files_data_iter = listdir(path_data_iter)
            for file_data in files_data_iter:
                path_file = join(path_data_iter, file_data)
                loss_iter_iter, acc_iter_iter,pred_iter_iter,truth_iter_iter = self.test_one_iter(path_file, name=file_data)
                loss_te += loss_iter_iter / len(files_data_iter) / len(X_list)
                acc_te += acc_iter_iter / len(files_data_iter) / len(X_list)
                if counter == 0:
                    preds = pred_iter_iter
                    truths = truth_iter_iter
                    counter += 1
                else:
                    preds = np.concatenate((preds, pred_iter_iter), axis=0)
                    truths = np.concatenate((truths, truth_iter_iter), axis=0)
        return loss_te, acc_te, preds, truths
        
    
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
                    loss_val, acc_val,preds,truths = self.test_all(self.opts.path_validation)
                    self.val_loss.append(loss_val)
                    self.val_acc.append(acc_val)
                    statement += " Loss_val: " + str(loss_val)
                    if self.opts.bool_kappa:
                        statement += " Kappa: " + str(self.quadratic_kappa(preds, truths))
                    if self.opts.bool_confusion:
                        print self.confusion_matrix(preds, truths)
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
            
            
                
