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

def Le_Net(layer, is_training, class_num, batch_size, name="Le_Net"):
    """
    This is the famous LeNet incarnation of the inception network.
    All the power is in the convs, so this is quite simple.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[5,6],[0,2],
                         [5,16],[0,2]]
    layer = general_conv(layer, is_training, architecture_conv, name=name)
    layer = deconv2d_w_bias(layer, 4, class_num, batch_size, name=name+"_deconv")
    return layer

def Alex_Net(layer, is_training, class_num, batch_size, name="Alex_Net"):
    """
    This is the famous AlexNet incarnation of the inception network.
    All the power is in the convs, so this is quite simple.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[11,96,4],[0,2],
                         [11,256],[0,2],
                         [3,384],[3,384],[3,256],[0,2]]
    layer = general_conv(layer, is_training, architecture_conv, name=name)
    layer = deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def VGG11_Net(layer, is_training, class_num, batch_size, name="VGG11_Net"):
    """
    This is the 11-layer VGG Network.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[3,64],[0,2],
                         [3,128],[0,2],
                         [3,256],[3,256],[0,2],
                         [3,512],[3,512],[0,2],
                         [3,512],[3,512],[0,2]]
    layer = general_conv(layer, is_training, architecture_conv, name=name)
    layer = deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def VGG13_Net(layer, is_training, class_num, batch_size, name="VGG13_Net"):
    """
    This is the 13-layer VGG Network.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[3,64],[3,64],[0,2],
                         [3,128],[3,128],[0,2],
                         [3,256],[3,256],[0,2],
                         [3,512],[3,512],[0,2],
                         [3,512],[3,512],[0,2]]
    layer = general_conv(layer, is_training, architecture_conv, name=name)
    layer = deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def VGG16_Net(layer, is_training, class_num, batch_size, name="VGG16_Net"):
    """
    This is the 16-layer VGG Network.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[3,64],[3,64],[0,2],
                         [3,128],[3,128],[0,2],
                         [3,256],[3,256],[3,256],[0,2],
                         [3,512],[3,512],[3,512],[0,2],
                         [3,512],[3,512],[3,512],[0,2]]
    layer = general_conv(layer, is_training, architecture_conv, name=name)
    layer = deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def VGG19_Net(layer, is_training, class_num, batch_size, name="VGG19_Net"):
    """
    This is the 19-layer VGG Network.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[3,64],[3,64],[0,2],
                         [3,128],[3,128],[0,2],
                         [3,256],[3,256],[3,256],[3,256],[0,2],
                         [3,512],[3,512],[3,512],[3,512],[0,2],
                         [3,512],[3,512],[3,512],[3,512],[0,2]]
    layer = general_conv(layer, is_training, architecture_conv, name=name)
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
    - layer: (tensor.4d) input tensor.
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

def Inception_Net(layer, is_training, class_num, batch_size, name="Inceptionv3_Net"):
    """
    This is the famous Inception v3 Network.
    This is a big big fucking network.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    # 224x224x?
    layer = conv2d_bn_relu(layer, is_training, 3, 32, stride=2, name=name+'_conv0')
    # 112x112x32
    layer = conv2d_bn_relu(layer, is_training, 3, 32, name=name+'_conv1')
    # 112x112x32
    layer = conv2d_bn_relu(layer, is_training, 3, 64, name=name+'_conv2')
    layer = max_pool(layer, k=3, stride=2)
    # 56x56x64
    layer = conv2d_bn_relu(layer, is_training, 1, 80, name=name+'_conv3')
    # 56x56x80
    layer = conv2d_bn_relu(layer, is_training, 3, 192, name=name+'_conv4')
    layer = max_pool(layer, k=3, stride=2)
    # 28x28x192
    branch1x1 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept1branch1')
    branch5x5 = conv2d_bn_relu(layer, is_training, 1, 48, name=name+'_incept1branch5a')
    branch5x5 = conv2d_bn_relu(branch5x5, is_training, 5, 64, name=name+'_incept1branch5b')
    branch3x3 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept1branch3a')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept1branch3b')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept1branch3c')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 32, name=name+'_incept1branchpool')
    layer = tf.concat([branch1x1, branch5x5, branch3x3, branchpool], 3)
    # 28x28x256
    branch1x1 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept2branch1')
    branch5x5 = conv2d_bn_relu(layer, is_training, 1, 48, name=name+'_incept2branch5a')
    branch5x5 = conv2d_bn_relu(branch5x5, is_training, 5, 64, name=name+'_incept2branch5b')
    branch3x3 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept2branch3a')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept2branch3b')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept2branch3c')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 64, name=name+'_incept2branchpool')
    layer = tf.concat([branch1x1, branch5x5, branch3x3, branchpool], 3)
    # 28x28x288
    branch1x1 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept3branch1')
    branch5x5 = conv2d_bn_relu(layer, is_training, 1, 48, name=name+'_incept3branch5a')
    branch5x5 = conv2d_bn_relu(branch5x5, is_training, 5, 64, name=name+'_incept3branch5b')
    branch3x3 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept3branch3a')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept3branch3b')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept3branch3c')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 64, name=name+'_incept3branchpool')
    layer = tf.concat([branch1x1, branch5x5, branch3x3, branchpool], 3)
    # 28x28x288
    branch1x1 = conv2d_bn_relu(layer, is_training, 3, 384, stride=2, name=name+'_incept4branch1')
    branch3x3 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept4branch3a')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept4branch3b')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, stride=2, name=name+'_incept4branch3c')
    branchpool = max_pool(layer, k=3, stride=2)
    layer = tf.concat([branch1x1, branch3x3, branchpool], 3)
    # 14x14x768
    branch1 = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept5branch1')
    branch7a = conv2d_bn_relu(layer, is_training, 1, 128, name=name+'_incept5branch7Aa')
    branch7a = conv2d_bn_relu(branch7a, is_training, [1, 7], 128, name=name+'_incept5branch7Ab')
    branch7a = conv2d_bn_relu(branch7a, is_training, [7, 1], 192, name=name+'_incept5branch7Ac')
    branch7b = conv2d_bn_relu(layer, is_training, 1, 128, name=name+'_incept5branch7Ba')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 128, name=name+'_incept5branch7Bb')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 128, name=name+'_incept5branch7Bc')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 128, name=name+'_incept5branch7Bd')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 192, name=name+'_incept5branch7Be')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 192, name=name+'_incept5branchpool')
    layer = tf.concat([branch1, branch7a, branch7b, branchpool], 3)
    # 14x14x768
    branch1 = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept6branch1')
    branch7a = conv2d_bn_relu(layer, is_training, 1, 160, name=name+'_incept6branch7Aa')
    branch7a = conv2d_bn_relu(branch7a, is_training, [1, 7], 160, name=name+'_incept6branch7Ab')
    branch7a = conv2d_bn_relu(branch7a, is_training, [7, 1], 192, name=name+'_incept6branch7Ac')
    branch7b = conv2d_bn_relu(layer, is_training, 1, 128, name=name+'_incept6branch7Ba')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 160, name=name+'_incept6branch7Bb')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 160, name=name+'_incept6branch7Bc')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 160, name=name+'_incept6branch7Bd')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 192, name=name+'_incept6branch7Be')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 192, name=name+'_incept6branchpool')
    layer = tf.concat([branch1, branch7a, branch7b, branchpool], 3)
    seg   = deconv2d_wo_bias(layer, 16, class_num, batch_size, name=name+"_incept6_deconv")
    # 14x14x768
    branch1 = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept7branch1')
    branch7a = conv2d_bn_relu(layer, is_training, 1, 160, name=name+'_incept7branch7Aa')
    branch7a = conv2d_bn_relu(branch7a, is_training, [1, 7], 160, name=name+'_incept7branch7Ab')
    branch7a = conv2d_bn_relu(branch7a, is_training, [7, 1], 192, name=name+'_incept7branch7Ac')
    branch7b = conv2d_bn_relu(layer, is_training, 1, 128, name=name+'_incept7branch7Ba')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 160, name=name+'_incept7branch7Bb')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 160, name=name+'_incept7branch7Bc')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 160, name=name+'_incept7branch7Bd')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 192, name=name+'_incept7branch7Be')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 192, name=name+'_incept7branchpool')
    layer = tf.concat([branch1, branch7a, branch7b, branchpool], 3)
    # 14x14x768
    branch1 = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept8branch1')
    branch7a = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept8branch7Aa')
    branch7a = conv2d_bn_relu(branch7a, is_training, [1, 7], 192, name=name+'_incept8branch7Ab')
    branch7a = conv2d_bn_relu(branch7a, is_training, [7, 1], 192, name=name+'_incept8branch7Ac')
    branch7b = conv2d_bn_relu(layer, is_training, 1, 128, name=name+'_incept8branch7Ba')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 192, name=name+'_incept8branch7Bb')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 192, name=name+'_incept8branch7Bc')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 192, name=name+'_incept8branch7Bd')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 192, name=name+'_incept8branch7Be')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 192, name=name+'_incept8branchpool')
    layer = tf.concat([branch1, branch7a, branch7b, branchpool], 3)
    # 14x14x768
    branch3x3 = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept9branch3a')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 320, stride=2, name=name+'_incept9branch3b')
    branch7x7 = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept9branch7a')
    branch7x7 = conv2d_bn_relu(branch7x7, is_training, [1, 7], 192, name=name+'_incept9branch7b')
    branch7x7 = conv2d_bn_relu(branch7x7, is_training, [7, 1], 192, name=name+'_incept9branch7c')
    branch7x7 = conv2d_bn_relu(branch7x7, is_training, 3, 192, stride=2, name=name+'_incept9branch7d')
    branchpool = max_pool(layer, k=3, stride=2)
    layer = tf.concat([branch3x3, branch7x7, branchpool], 3)
    # 7x7x1280
    branch1 = conv2d_bn_relu(layer, is_training, 1, 320, name=name+'_inceptAbranch1')
    branch3a = conv2d_bn_relu(layer, is_training, 1, 384, name=name+'_inceptAbranch3Aa')
    branch3a = tf.concat([conv2d_bn_relu(branch3a, is_training, [1, 3], 384, name=name+'_inceptAbranch3Ab'),
                             conv2d_bn_relu(branch3a, is_training, [3, 1], 384, name=name+'_inceptAbranch3Ac')], 3)
    branch3b = conv2d_bn_relu(layer, is_training, 1, 448, name=name+'_inceptAbranch3Ba')
    branch3b = conv2d_bn_relu(branch3b, is_training, 3, 384, name=name+'_inceptAbranch3Bb')
    branch3b = tf.concat([conv2d_bn_relu(branch3b, is_training, [1, 3], 384, name=name+'_inceptAbranch3Bc'),
                             conv2d_bn_relu(branch3b, is_training, [3, 1], 384, name=name+'_inceptAbranch3Bd')], 3)
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 192, name=name+'_inceptAbranchpool')
    layer = tf.concat([branch1, branch3a, branch3b, branchpool], 3)
    # 7x7x2048
    branch1 = conv2d_bn_relu(layer, is_training, 1, 320, name=name+'_inceptBbranch1')
    branch3a = conv2d_bn_relu(layer, is_training, 1, 384, name=name+'_inceptBbranch3Aa')
    branch3a = tf.concat([conv2d_bn_relu(branch3a, is_training, [1, 3], 384, name=name+'_inceptBbranch3Ab'),
                             conv2d_bn_relu(branch3a, is_training, [3, 1], 384, name=name+'_inceptBbranch3Ac')], 3)
    branch3b = conv2d_bn_relu(layer, is_training, 1, 448, name=name+'_inceptBbranch3Ba')
    branch3b = conv2d_bn_relu(branch3b, is_training, 3, 384, name=name+'_inceptBbranch3Bb')
    branch3b = tf.concat([conv2d_bn_relu(branch3b, is_training, [1, 3], 384, name=name+'_inceptBbranch3Bc'),
                             conv2d_bn_relu(branch3b, is_training, [3, 1], 384, name=name+'_inceptBbranch3Bd')], 3)
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 192, name=name+'_inceptBbranchpool')
    layer = tf.concat([branch1, branch3a, branch3b, branchpool], 3)
    seg   += deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_inceptB_deconv")
    return seg

def conv_res(layer, is_training, architecture=[[1, 64], [3, 64], [1, 256]], alpha=0.1, name="conv_res"):
    """
    This is going to be a residual layer.
    We do 3 convolutions and add to the original input.
    INPUTS:
    - layer: (tensor.4d) input tensor
    - is_training: (variable) whether or not we're training
    - architecture: (list of lists) architecture of 3 convs
    - alpha: (float) for the relu
    - name: (string) name of the layer
    """
    l_input = layer #save for later
    for iter_num, kSize in enumerate(architecture):
        layer = batch_norm(layer, is_training, name=(name+'_bn'+str(iter_num)))
        layer = tf.maximum(layer, layer*alpha)
        layer = conv2d_wo_bias(layer, kSize[0], kSize[1], name=(name+"_conv2d"+str(iter_num)))
    if l_input.get_shape().as_list()[3] != kSize[1]:
        l_input = tf.pad(l_input, [[0,0],[0,0],[0,0],[0,kSize[1]-l_input.get_shape().as_list()[3]]])
    layer += l_input
    return layer

def Res_Net(layer, is_training, class_num, batch_size, name="Res_Net"):
    """
    This is the famous Res Net.
    150+ Layers mother fucker!  Fuck that shit..
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    layer = conv2d_wo_bias(layer, 7, 64, stride=2, name=name+"_conv1")
    layer = max_pool(layer, k=3, stride=2)
    for i in range(3):
        layer = conv_res(layer, is_training, architecture=[[1,64],[3,64],[1,256]], name=name+"_conv2_"+str(i))
    layer = max_pool(layer, k=3, stride=2)
    for i in range(8):
        layer = conv_res(layer, is_training, architecture=[[1,128],[3,128],[1,512]], name=name+"_conv3_"+str(i))
    layer = max_pool(layer, k=3, stride=2)
    seg   = deconv2d_wo_bias(layer, 16, class_num, batch_size, name=name+"_covn4_deconv")
    for i in range(36):
        layer = conv_res(layer, is_training, architecture=[[1,256],[3,256],[1,1024]], name=name+"_conv4_"+str(i))
    layer = max_pool(layer, k=3, stride=2)
    for i in range(3):
        layer = conv_res(layer, is_training, architecture=[[1,512],[3,512],[1,2048]], name=name+"_conv5_"+str(i))
    seg   += deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_conv5_deconv")
    return seg
