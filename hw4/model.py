# %load model.py
# %load model.py
import os
import time
import numpy as np
import tensorflow as tf

class VAE():
    def __init__(self):
        super().__init__()

    def load(self, npy_path):
        self.data_dict = np.load(npy_path, encoding='latin1').item()
        print("Load %s as self.data_dict" % npy_path)

    def build(self, n_dim=512, lambda_KL=1e-5, batch_size = 64, shape=(64,64,3)):
        """
        load pre-trained weights from path
        :param vgg16_npy_path: file path of vgg16 pre-trained weights
        """
        # input information
        self.H, self.W, self.C = shape
        self.batch_size = batch_size
        self.n_dim = n_dim
        self.lambda_KL = lambda_KL
        # parameter dictionary
        self.para_dict = dict()
        self.data_dict = dict()
        self.net_shape = dict()

        # input placeholder
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, self.C])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, self.C])
        self.is_train = tf.placeholder(tf.bool)
        
        # normalize inputs
        self.x = (self.x-128.0)/128.0
        assert self.x.get_shape().as_list()[1:] == [self.H, self.W, self.C]

        # conv
        conv1 = self.conv_bn_layer(self.x, shape=(4,4,3,32), stride=2, name="conv1")
        conv2 = self.conv_bn_layer(conv1 , shape=(4,4,32,64), stride=2, name="conv2")
        conv3 = self.conv_bn_layer(conv2 , shape=(4,4,64,128), stride=2, name="conv3")
        conv4 = self.conv_bn_layer(conv3 , shape=(4,4,128,256), stride=2, name="conv4")
        flatten = self.flatten_layer(conv4, name='flatten')

        # mean and logvar
        mean = self.dense_layer(flatten, n_hidden=self.n_dim, name='mean')
        logvar = self.dense_layer(flatten, n_hidden=self.n_dim, name='logvar')
        
        # sample
        epsilon = tf.cond(self.is_train,
                            lambda:tf.random_normal(shape=logvar.get_shape()), 
                            lambda:tf.zeros(shape=logvar.get_shape()))
        
        # mean + eps * sd, where sd = exp(0.5*logvar)
        sample_input = tf.add(mean, tf.multiply(epsilon, tf.exp(0.5*logvar)))

        # deconv
        deconv_fc1 = self.dense_layer(sample_input, n_hidden=self.net_shape['flatten'][1], name='deconv_fc1')
        deconv_input = tf.reshape(deconv_fc1, shape=self.net_shape['conv4'])
        deconv1 = self.trans_conv_layer(bottom=deconv_input, shape=(4,4,128,256),
                                        output_shape=self.net_shape['conv3'], stride=2, name='deconv1')
        deconv2 = self.trans_conv_layer(bottom=deconv1, shape=(4,4,64,128),
                                        output_shape=self.net_shape['conv2'], stride=2, name='deconv2')
        deconv3 = self.trans_conv_layer(bottom=deconv2, shape=(4,4,32,64),
                                        output_shape=self.net_shape['conv1'], stride=2, name='deconv3')
        self.output = self.trans_conv_layer(bottom=deconv3, shape=(4,4,3,32),
                                            output_shape=(self.batch_size, self.H, self.W, self.C), activation='tanh', stride=2, name='deconv_output')

        reconstruction_loss = tf.losses.mean_squared_error(labels=self.x,predictions=self.output)
        KL_loss = -0.5*(tf.subtract(tf.add(1.0, logvar), tf.add(tf.square(mean), tf.exp(logvar))))
        self.loss = dict()
        self.loss['reconstruction'] = reconstruction_loss
        self.loss['KL_loss'] = KL_loss
        self.train_op = tf.reduce_mean(reconstruction_loss) + self.lambda_KL*tf.reduce_mean(KL_loss)

    def dense_layer(self, bottom, n_hidden=None, name=None):
        bottom_shape = bottom.get_shape().as_list()
        with tf.variable_scope("VAE", reuse=tf.AUTO_REUSE):
            if n_hidden is not None:
                W = self.get_weights(shape=(bottom_shape[1], n_hidden), name=name)
                b = self.get_bias(shape=n_hidden, name=name)
            elif name in self.data_dict.keys():
                W = self.get_weights(name=name)
                b = self.get_bias(name=name)
            else:
                print("Neither give a shape nor lack a pre-trained layer called %s" % name)
        self.para_dict[name] = [W, b]
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        self.net_shape[name] = fc.get_shape().as_list()
        return fc

    def flatten_layer(self, bottom, name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        flatten = tf.reshape(bottom, [-1, dim])
        self.net_shape[name] = flatten.get_shape().as_list()
        return flatten

    def avg_pool_layer(self, bottom, name):
        pool = tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        self.net_shape[name] = pool.get_shape().as_list()
        return pool

    def max_pool_layer(self, bottom, name):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        self.net_shape[name] = pool.get_shape().as_list()
        return pool
    
    def dropout(self, bottom, keep_prob):
        if self.is_train == True:
            return tf.nn.dropout(bottom, keep_prob=keep_prob)
        else:
            return bottom

    def trans_conv_layer(self, bottom, output_shape, stride, activation='relu', name=None, shape=None):
        with tf.variable_scope("VAE", reuse=tf.AUTO_REUSE):
            if shape is not None:
                conv_filter, gamma, beta, bn_mean, bn_variance = self.get_conv_filter(shape=shape, name=name)
                conv_bias = self.get_bias(shape=shape[2], name=name)
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

        conv = tf.nn.conv2d_transpose(bottom, conv_filter, output_shape, strides=[1, stride, stride, 1], padding="SAME")
        conv = tf.nn.bias_add(conv, conv_bias)
        
        from tensorflow.python.training.moving_averages import assign_moving_average
        def mean_var_with_update():
            mean, variance = tf.nn.moments(conv, [0,1,2], name='moments')
            with tf.control_dependencies([assign_moving_average(bn_mean, mean, 0.99),
                                            assign_moving_average(bn_variance, variance, 0.99)]):
                return tf.identity(mean), tf.identity(variance)

        mean, variance = tf.cond(self.is_train, mean_var_with_update, lambda:(bn_mean, bn_variance))
        conv = tf.nn.batch_normalization(conv, mean, variance, beta, gamma, 1e-05)
        self.net_shape[name] = conv.get_shape().as_list()

        if activation=='tanh':
            tanh = tf.nn.tanh(conv)
            return tanh
        else:
            relu = tf.nn.relu(conv)
            return relu

    def conv_bn_layer(self, bottom, stride=1, activation='relu', name=None, shape=None):
        with tf.variable_scope("VAE",reuse=tf.AUTO_REUSE):
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

        conv = tf.nn.conv2d(bottom, conv_filter, [1, stride, stride, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, conv_bias)

        from tensorflow.python.training.moving_averages import assign_moving_average
        def mean_var_with_update():
            mean, variance = tf.nn.moments(conv, [0,1,2], name='moments')
            with tf.control_dependencies([assign_moving_average(bn_mean, mean, 0.99),
                                            assign_moving_average(bn_variance, variance, 0.99)]):
                return tf.identity(mean), tf.identity(variance)

        mean, variance = tf.cond(self.is_train, mean_var_with_update, lambda:(bn_mean, bn_variance))

        conv = tf.nn.batch_normalization(conv, mean, variance, beta, gamma, 1e-05)
        self.net_shape[name] = conv.get_shape().as_list()

        if activation=='tanh':
            tanh = tf.nn.tanh(conv)
            return tanh
        else:
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
            if 'deconv' in name:
                H,W,O,C = conv_filter.get_shape().as_list()
            else:
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
    
    def get_weights(self, shape=None, name=None):
        if shape is not None:
            return tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name=name+"_W", dtype=tf.float32)
        elif name in self.data_dict.keys(): 
            return tf.get_variable(initializer=self.data_dict[name][0], name=name+"_W")
        else:
            print("(get_weight) neither give a shape nor lack a pre-trained layer called %s" % name)
            return None
            
    def get_bias(self, shape=None, name=None):
        if shape is not None:
            return tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name=name+"_b", dtype=tf.float32)
        elif name in self.data_dict.keys(): 
            return tf.get_variable(initializer=self.data_dict[name][1], name=name+"_b")
        else:
            print("(get_bias) neither give a shape nor lack a pre-trained layer called %s" % name)
            return None
