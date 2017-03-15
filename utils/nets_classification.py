import tensorflow as tf

from layers import *

def general_conv(layer, is_training, architecture_conv, name="general_conv"):
    """
    A generalized convolution block that takes an architecture.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - is_training: (bool) are we in training size
    - architecture_conv: (list of lists)
      [[filt_size, filt_num, stride], ..., [0, poolSize],
       [filt_size, filt_num, stride], ..., [0, poolSize],
       ...]
    - b_name: (string) branch name.  If not doing branch, doesn't matter.
    """
    for conv_iter, conv_numbers in enumerate(architecture_conv):
        if conv_numbers[0]==0:
            layer = max_pool(layer, k=conv_numbers[1])
        else:
            if len(conv_numbers)==2:
                conv_numbers.append(1)
            layer = conv2d_bn_relu(layer, is_training, conv_numbers[0], conv_numbers[1], stride=conv_numbers[2],
                           name=(name+"_conv"+str(conv_iter)))
    return layer

def Le_Net(X, is_training, class_num, batch_size, keep_prob=1.0, name="Le_Net"):
    """
    This is the famous LeNet incarnation of the inception network.
    All the power is in the convs, so this is quite simple.
    INPUTS:
    - X: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[5,6],[0,2],
                         [5,16],[0,2]]
    layer = general_conv(X, is_training, architecture_conv, name=name)
    layer = dense_bn_do_relu(layer, is_training, 500, keep_prob, name=name+'_hidden')
    layer = dense_w_bias(layer, class_num, name=name+'_output')
    return layer

def Alex_Net(X, is_training, class_num, batch_size, keep_prob=1.0, name="Alex_Net"):
    """
    This is the famous AlexNet incarnation of the inception network.
    All the power is in the convs, so this is quite simple.
    INPUTS:
    - X: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[11,96],[0,2],
                         [11,256],[0,2],
                         [3,384],[3,384],[3,256],[0,2]]
    layer = general_conv(X, is_training, architecture_conv, name=name)
    layer = dense_bn_do_relu(layer, is_training, 4096, keep_prob, name=name+'_hidden1')
    layer = dense_bn_do_relu(layer, is_training, 4096, keep_prob, name=name+'_hidden2')
    layer = dense_w_bias(layer, class_num, name=name+'_output')
    return layer

def VGG11_Net(X, is_training, class_num, batch_size, name="VGG11_Net"):
    """
    This is the 11-layer VGG Network.
    INPUTS:
    - X: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[3,64],[0,2],
                         [3,128],[0,2],
                         [3,256],[3,256],[0,2],
                         [3,512],[3,512],[0,2],
                         [3,512],[3,512],[0,2]]
    layer = general_conv(X, is_training, architecture_conv, name=name)
    layer = deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def VGG13_Net(X, is_training, class_num, batch_size, name="VGG13_Net"):
    """
    This is the 13-layer VGG Network.
    INPUTS:
    - X: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[3,64],[3,64],[0,2],
                         [3,128],[3,128],[0,2],
                         [3,256],[3,256],[0,2],
                         [3,512],[3,512],[0,2],
                         [3,512],[3,512],[0,2]]
    layer = general_conv(X, is_training, architecture_conv, name=name)
    layer = deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def VGG16_Net(X, is_training, class_num, batch_size, name="VGG16_Net"):
    """
    This is the 16-layer VGG Network.
    INPUTS:
    - X: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[3,64],[3,64],[0,2],
                         [3,128],[3,128],[0,2],
                         [3,256],[3,256],[3,256],[0,2],
                         [3,512],[3,512],[3,512],[0,2],
                         [3,512],[3,512],[3,512],[0,2]]
    layer = general_conv(X, is_training, architecture_conv, name=name)
    layer = deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def VGG19_Net(X, is_training, class_num, batch_size, name="VGG19_Net"):
    """
    This is the 19-layer VGG Network.
    INPUTS:
    - X: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[3,64],[3,64],[0,2],
                         [3,128],[3,128],[0,2],
                         [3,256],[3,256],[3,256],[3,256],[0,2],
                         [3,512],[3,512],[3,512],[3,512],[0,2],
                         [3,512],[3,512],[3,512],[3,512],[0,2]]
    layer = general_conv(X, is_training, architecture_conv, name=name)
    layer = deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def inceptionv1_module(layer, is_training, kSize=[16,16,16,16,16,16], name="inceptionv1_module"):
    """
    So, this is the classical incept layer.
    INPUTS:
    - layer: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - is_training: (bool) are we in training size
    - ksize: (array (6,)) [1x1, 3x3reduce, 3x3, 5x5reduce, 5x5, poolproj]
    - name: (string) name of incept layer
    """
    layer_1x1 = conv2d_bn_relu(layer, is_training, 1, kSize[0], name=(name+"_1x1"))
    layer_3x3a = conv2d_bn_relu(layer, is_training, 1, kSize[1], name=(name+"_3x3a"))
    layer_3x3b = conv2d_bn_relu(layer_3x3a, is_training, 3, kSize[2], name=(name+"_3x3b"))
    layer_5x5a = conv2d_bn_relu(layer, is_training, 1, kSize[3], name=(name+"_5x5a"))
    layer_5x5b = conv2d_bn_relu(layer_5x5a, is_training, 5, kSize[4], name=(name+"_5x5b"))
    layer_poola = max_pool(layer, k=3, stride=1)
    layer_poolb = conv2d_bn_relu(layer_poola, is_training, 1, kSize[5], name=(name+"_poolb"))
    return tf.concat([layer_1x1, layer_3x3b, layer_5x5b, layer_poolb], 3)

def GoogLe_Net(layer, is_training, class_num, batch_size, name="GoogLe_Net"):
    """
    This is the famous GoogLeNet incarnation of the inception network.
    All the power is in the convs, so this is quite simple.
    INPUTS:
    - X: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    # Conv1
    layer = conv2d_bn_relu(layer, is_training, 7, 64, stride=2, name=name+"_conv1")
    layer = max_pool(layer, k=3, stride=2)
    # Conv2
    layer = conv2d_bn_relu(layer, is_training, 1, 64, name=name+"_conv2a")
    layer = conv2d_bn_relu(layer, is_training, 3, 192, name=name+"_conv2b")
    layer = max_pool(layer, k=3, stride=2)
    # Incept3
    layer = inceptionv1_module(layer, is_training, kSize=[64,96,128,16,32,32], name=name+"_incept3a")
    layer = inceptionv1_module(layer, is_training, kSize=[128,128,192,32,96,64], name=name+"_incept3b")
    layer = max_pool(layer, k=3, stride=2)
    # Incept4
    layer = inceptionv1_module(layer, is_training, kSize=[192,96,208,16,48,64], name=name+"_incept4a")
    layer = inceptionv1_module(layer, is_training, kSize=[160,112,224,24,64,64], name=name+"_incept4b")
    seg   = deconv2d_wo_bias(layer, 16, class_num, batch_size, name=name+"_incept4b_deconv")
    layer = inceptionv1_module(layer, is_training, kSize=[128,128,256,24,64,64], name=name+"_incept4c")
    layer = inceptionv1_module(layer, is_training, kSize=[112,144,288,32,64,64], name=name+"_incept4d")
    layer = inceptionv1_module(layer, is_training, kSize=[256,160,320,32,128,128], name=name+"_incept4e")
    layer = max_pool(layer, k=3, stride=2)
    # Incept5
    layer = inceptionv1_module(layer, is_training, kSize=[256,160,320,32,128,128], name=name+"_incept5a")
    layer = inceptionv1_module(layer, is_training, kSize=[384,192,384,48,128,128], name=name+"_incept5b")
    seg   += deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_incept5b_deconv")
    return seg
