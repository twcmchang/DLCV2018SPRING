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

    def build(self, n_dim=512, lambda_kl=1e-5, shape=(64,64,3)):
        """
        load pre-trained weights from path
        :param vgg16_npy_path: file path of vgg16 pre-trained weights
        """
        # input information
        self.H, self.W, self.C = shape
        self.n_dim = n_dim
        self.lambda_kl = lambda_kl
        
        # parameter dictionary
        self.para_dict = dict()
        self.data_dict = dict()
        self.net_shape = dict()

        # input placeholder
        self.x = tf.placeholder(tf.float32, [None, self.H, self.W, self.C])
        self.is_train = tf.placeholder(tf.bool)
        self.random_sample = tf.placeholder(tf.float32, [None, self.n_dim])
        
        # normalize inputs
        # self.x = self.x/255.0
        assert self.x.get_shape().as_list()[1:] == [self.H, self.W, self.C]
        
        with tf.variable_scope("VAE",reuse=tf.AUTO_REUSE):
            self.mean, self.logvar = self.encoder(self.x)
            self.sample = self.sampler(self.mean, self.logvar)
            self.output_image = self.decoder(self.sample)
            
            self.recon_loss = tf.reduce_mean(tf.square(tf.subtract(self.x, self.output_image)), [1,2,3])
            self.kl_loss = 0.5*tf.reduce_mean(tf.subtract(tf.add(tf.square(self.mean), tf.exp(self.logvar)), tf.add(1.0, self.logvar)),1)
            self.vae_loss = self.recon_loss + self.lambda_kl*self.kl_loss
            
            # Sampling from random z
            self.random_sample_images = self.decoder(self.random_sample)
    
    def encoder(self, input_image):
        # conv
        conv1 = self.conv_bn_layer(input_image, shape=(4,4,3,32), stride=2, name="conv1")
        conv2 = self.conv_bn_layer(conv1 , shape=(4,4,32,64), stride=2, name="conv2")
        conv3 = self.conv_bn_layer(conv2 , shape=(4,4,64,128), stride=2, name="conv3")
        conv4 = self.conv_bn_layer(conv3 , shape=(4,4,128,256), stride=2, name="conv4")
        flatten = self.flatten_layer(conv4, name='flatten')

        # mean and logvar
        mean = self.dense_layer(flatten, n_hidden=self.n_dim, name='mean')
        logvar = self.dense_layer(flatten, n_hidden=self.n_dim, name='logvar')
        return mean, logvar

    def sampler(self, mean, logvar):
        eps = tf.random_normal(shape=tf.shape(mean))
        return mean + tf.exp(logvar / 2) * eps
        
    def decoder(self, sample_input):
        deconv_fc1 = self.dense_layer(sample_input, n_hidden=self.net_shape['flatten'][1], name='deconv_fc1')
        deconv_input = tf.reshape(deconv_fc1, shape=[-1, 4, 4, 256])
        
        batch_size = tf.shape(sample_input)[0]
        
        deconv1 = self.trans_conv_layer(bottom=deconv_input, shape=(4,4,128,256),
                                        output_shape=[batch_size, 8, 8, 128], stride=2, name='deconv1')
        deconv2 = self.trans_conv_layer(bottom=deconv1, shape=(4,4,64,128),
                                        output_shape=[batch_size, 16, 16, 64], stride=2, name='deconv2')
        deconv3 = self.trans_conv_layer(bottom=deconv2, shape=(4,4,32,64),
                                        output_shape=[batch_size, 32, 32, 32], stride=2, name='deconv3')
        output = self.trans_conv_layer(bottom=deconv3, shape=(4,4,3,32),
                                        output_shape=[batch_size, self.H, self.W, self.C], activation='tanh', stride=2, name='deconv_output')
        return (output/2) + 0.5

    def dense_layer(self, bottom, n_hidden=None, name=None):
        bottom_shape = bottom.get_shape().as_list()
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
            print('tanh')
            tanh = tf.nn.tanh(conv)
            return tanh
        else:
            relu = tf.nn.leaky_relu(conv)
            return relu

    def conv_bn_layer(self, bottom, stride=1, activation='lrelu', name=None, shape=None):
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
            relu = tf.nn.leaky_relu(conv)
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
                beta = tf.get_variable(initializer=self.data_dict[name+"_beta"], name=name+"_beta")
            else:
                beta = tf.get_variable(shape=(O,), initializer=tf.zeros_initializer(), name=name+'_beta')

            if name+"_bn_mean" in self.data_dict.keys(): 
                bn_mean = tf.get_variable(initializer=self.data_dict[name+"_bn_mean"], name=name+"_bn_mean")
            else:
                bn_mean = tf.get_variable(shape=(O,), initializer=tf.zeros_initializer(), name=name+'_bn_mean')

            if name+"_bn_variance" in self.data_dict.keys(): 
                bn_variance = tf.get_variable(initializer=self.data_dict[name+"_bn_variance"], name=name+"_bn_variance")
            else:
                bn_variance = tf.get_variable(shape=(O,), initializer=tf.ones_initializer(), name=name+'_bn_variance')
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
