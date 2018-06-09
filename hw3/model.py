# %load model.py
import os
import time
import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68] # [B, G, R]
class VGG16:
    def __init__(self, classes=7, shape=(256,256,3)):
        # input information
        self.H, self.W, self.C = shape
        self.classes = classes

        # parameter dictionary
        self.para_dict = dict()
        
    def build(self, vgg16_npy_path, mode='FCN32s', keep_prob=1.0):
        """
        load pre-trained weights from path
        :param vgg16_npy_path: file path of vgg16 pre-trained weights
        """

        # input placeholder
        self.x = tf.placeholder(tf.float32, [None, self.H, self.W, self.C])
        self.y = tf.placeholder(tf.int64, [None, self.H, self.W, self.classes])
        self.is_train = tf.placeholder(tf.bool)
        
        # normalize inputs
        self.x = self.x/255.0
        assert self.x.get_shape().as_list()[1:] == [self.H, self.W, self.C]

        # load pre-trained weights
        if isinstance(vgg16_npy_path,dict):
            self.data_dict = vgg16_npy_path
            print("parameters loaded")
        else:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            print("npy file loaded")
        
        self._keep_prob = keep_prob

        ### pre-trained VGG-16 start ###
        conv1_1 = self.conv_bn_layer( self.x, "conv1_1")
        conv1_2 = self.conv_bn_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, "pool1")

        conv2_1 = self.conv_bn_layer(  pool1, "conv2_1")
        conv2_2 = self.conv_bn_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, "pool2")

        conv3_1 = self.conv_bn_layer(  pool2, "conv3_1")
        conv3_2 = self.conv_bn_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_bn_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, "pool3")

        conv4_1 = self.conv_bn_layer(  pool3, "conv4_1")
        conv4_2 = self.conv_bn_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_bn_layer(conv4_2, "conv4_3")
        pool4   = self.max_pool(conv4_3, "pool4")

        conv5_1 = self.conv_bn_layer(  pool4, "conv5_1")
        conv5_2 = self.conv_bn_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_bn_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, "pool5")
        ### pre-trained VGG-16 end ###
    
        ### convert to fully convolutional layer ###
        conv6 = self.conv_bn_layer(pool5, name="conv6", shape=(1, 1, 512, 4096))
        conv6 = self.dropout_layer(conv6, keep_prob=self._keep_prob)

        conv7 = self.conv_bn_layer(conv6, name="conv7", shape=(1, 1, 4096, 4096))
        conv7 = self.dropout_layer(conv7, keep_prob=self._keep_prob)

        conv8 = self.conv_bn_layer(conv7, name="conv8", shape=(1, 1, 4096, self.classes))

        ### transpose convolutional layer ###
        ### FCN8s
        if mode=="FCN32s":
            logits = self.trans_conv_layer(bottom=conv8, shape=(64, 64, self.classes, self.classes), # (h, w, out, in)
                                           output_shape=(tf.shape(self.x)[0], self.H, self.W, self.classes), stride=32, name="logits")
        elif mode=="FCN16s":
            shape1 = pool4.get_shape().as_list()
            trans_conv1 = self.trans_conv_layer(bottom=conv8, shape=(4, 4, shape1[3], self.classes), # (h, w, out, in)
                                                output_shape=tf.shape(pool4), stride=2, name="trans_conv1")
            fuse1 = tf.add(trans_conv1, pool4, name="fuse1")
            
            #shapeX = tf.shape(self.x)
            logits = self.trans_conv_layer(bottom=fuse1, shape=(32, 32, self.classes, shape1[3]), # (h, w, out, in)
                                           output_shape=(tf.shape(self.x)[0], self.H, self.W, self.classes), stride=16, name="logits")
        elif mode=="FCN8s":
            shape1 = pool4.get_shape().as_list()
            trans_conv1 = self.trans_conv_layer(bottom=conv8, shape=(4, 4, shape1[3], self.classes), 
                                                output_shape=tf.shape(pool4), stride=2, name="trans_conv1")
            fuse1 = tf.add(trans_conv1, pool4, name="fuse1")

            shape2 = pool3.get_shape().as_list()
            trans_conv2 = self.trans_conv_layer(bottom=fuse1, shape=(4, 4, shape2[3], shape1[3]), 
                                                output_shape=tf.shape(pool3), stride=2, name="trans_conv2")
            fuse2 = tf.add(trans_conv2, pool3, name="fuse2")

            logits = self.trans_conv_layer(bottom=fuse2, shape=(16, 16, self.classes, shape2[3]), 
                                           output_shape=(tf.shape(self.x)[0], self.H, self.W, self.classes), stride=8, name="logits")
            ### transpose end ###
        self.pred = tf.argmax(logits, axis=3, name="pred")

        def train_operation():
            self.actual = tf.argmax(self.y, axis=3, name="answer")
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(x=self.pred, y=self.actual), tf.float32))
            return True

        _ = tf.cond(self.is_train, train_operation, lambda: False)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def dropout_layer(self, bottom, keep_prob):
        if self.is_train == True:
            return tf.nn.dropout(bottom, keep_prob=keep_prob)
        else:
            return bottom

    def trans_conv_layer(self, bottom, output_shape, stride, name=None, shape=None):
        with tf.variable_scope("VGG16", reuse=tf.AUTO_REUSE):
            if shape is not None:
                conv_filter = self.get_conv_filter(shape=shape, name=name, with_bn=False)
                conv_bias = self.get_bias(shape=shape[2], name=name)
            elif name in self.data_dict.keys():
                conv_filter = self.get_conv_filter(name=name, with_bn=False)
                conv_bias = self.get_bias(name=name)
            else:
                print("Neither give a shape nor lack a pre-trained layer called %s" % name)
        
        self.para_dict[name] = [conv_filter, conv_bias]

        conv = tf.nn.conv2d_transpose(bottom, conv_filter, output_shape, strides=[1, stride, stride, 1], padding="SAME")
        conv = tf.nn.bias_add(conv, conv_bias)

        return conv

    def conv_bn_layer(self, bottom, name=None, shape=None):
        with tf.variable_scope("VGG16",reuse=tf.AUTO_REUSE):
            if shape is not None:
                conv_filter, gamma, beta, bn_mean, bn_variance = self.get_conv_filter(shape=shape, name=name)
                conv_bias = self.get_bias(shape=shape[3], name=name)
            elif name in self.data_dict.keys():
                conv_filter, gamma, beta, bn_mean, bn_variance = self.get_conv_filter(name=name)
                conv_bias = self.get_bias(name=name)
            else:
                print("Neither give a shape nor lack a pre-trained layer called %s" % name)
        
        self.para_dict[name] = [conv_filter, conv_bias]
        self.para_dict[name+"_gamma"] = gamma
        self.para_dict[name+"_beta"] = beta
        self.para_dict[name+"_bn_mean"] = bn_mean
        self.para_dict[name+"_bn_variance"] = bn_variance

        conv = tf.nn.conv2d(bottom, conv_filter, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, conv_bias)

        from tensorflow.python.training.moving_averages import assign_moving_average
        def mean_var_with_update():
            mean, variance = tf.nn.moments(conv, [0,1,2], name='moments')
            with tf.control_dependencies([assign_moving_average(bn_mean, mean, 0.99),
                                            assign_moving_average(bn_variance, variance, 0.99)]):
                return tf.identity(mean), tf.identity(variance)

        mean, variance = tf.cond(self.is_train, mean_var_with_update, lambda:(bn_mean, bn_variance))

        conv = tf.nn.batch_normalization(conv, mean, variance, beta, gamma, 1e-05)
        relu = tf.nn.relu(conv)
        return relu

    def get_conv_filter(self, shape=None, name=None, with_bn=True):
        if shape is not None:
            conv_filter = tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name=name+"_W", dtype=tf.float32)
        elif name in self.data_dict.keys():
            conv_filter = tf.get_variable(initializer=self.data_dict[name][0], name=name+"_W")
        else:
            print("Neither give a shape nor lack a pre-trained layer called %s" % name)
            return None

        if with_bn:
            H,W,C,O = conv_filter.get_shape().as_list()

            if name+"_gamma" in self.data_dict.keys(): 
                gamma = tf.get_variable(initializer=self.data_dict[name+"_gamma"], name=name+"_gamma")
            else:
                gamma = tf.get_variable(shape=(O,), initializer=tf.ones_initializer(), name=name+"_gamma")

            if name+"_beta" in self.data_dict.keys(): 
                beta = tf.get_variable(initializer=self.data_dict[name+"_beta"], name=name+"_beta", trainable=False)
            else:
                beta = tf.get_variable(shape=(O,), initializer=tf.zeros_initializer(), name=name+'_beta')

            if name+"_bn_mean" in self.data_dict.keys(): 
                bn_mean = tf.get_variable(initializer=self.data_dict[name+"_bn_mean"], name=name+"_bn_mean", trainable=False)
            else:
                bn_mean = tf.get_variable(shape=(O,), initializer=tf.zeros_initializer(), name=name+'_bn_mean', trainable=False)

            if name+"_bn_variance" in self.data_dict.keys(): 
                bn_variance = tf.get_variable(initializer=self.data_dict[name+"_bn_variance"], name=name+"_bn_variance", trainable=False)
            else:
                bn_variance = tf.get_variable(shape=(O,), initializer=tf.ones_initializer(), name=name+'_bn_variance', trainable=False)
            return conv_filter, gamma, beta, bn_mean, bn_variance
        else:
            return conv_filter
            
    def get_bias(self, shape=None, name=None):
        if shape is not None:
            return tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name=name+"_b", dtype=tf.float32)
        elif name in self.data_dict.keys(): 
            return tf.get_variable(initializer=self.data_dict[name][1], name=name+"_b")
        else:
            print("(get_bias) neither give a shape nor lack a pre-trained layer called %s" % name)
            return None
