# %load model.py
import tensorflow as tf

class Model(object):
    def __init__(self, scope_name=""):
        super().__init__()
        self.scope_name = scope_name
        self.weight_decay = 0.0
        self.para_dict = dict()
        self.data_dict = dict()
    
    def avg_pool_layer(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool_layer(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def dropout_layer(self, bottom, keep_prob, is_train):
        return tf.cond(is_train, lambda: tf.nn.dropout(bottom, keep_prob=keep_prob),lambda:bottom)

    def conv_layer(self, bottom, name=None, shape=None):
        if shape is not None:
            conv_filter = tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name=name+"_W", dtype=tf.float32)
            conv_bias = tf.get_variable(shape=shape[2], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name=name+"_b", dtype=tf.float32)
        elif name in self.data_dict.keys():
            conv_filter = tf.get_variable(initializer=self.data_dict[name][0], name=name+"_W")
            conv_bias = tf.get_variable(initializer=self.data_dict[name][1], name=name+"_b")
        else:
            print("Neither give a shape nor lack a pre-trained layer called %s" % name)
            return None
        self.para_dict[name] = [conv_filter, conv_bias]
        self.weight_decay += tf.nn.l2_loss(conv_filter) + tf.nn.l2_loss(conv_bias)

        conv = tf.nn.conv2d(bottom, conv_filter, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, conv_bias)
        relu = tf.nn.relu(conv)
        return relu
    
    def flatten_layer(self, bottom):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])
        return x
    
    def dense_layer(self, bottom, shape=None, name=None):
        if shape is not None:
            weights = tf.get_variable(shape=shape, initializer=tf.random_normal_initializer(mean=0, stddev=0.02), name=name+"_W", dtype=tf.float32)
            biases = tf.get_variable(shape=shape[1], initializer=tf.constant_initializer(0.0), name=name+"_b", dtype=tf.float32)
        elif name in self.data_dict.keys():
            weights = tf.get_variable(initializer=self.data_dict[name][0], name=name+"_W")
            biases = tf.get_variable(initializer=self.data_dict[name][1], name=name+"_b")
        else:
            print("Neither give a shape nor lack a pre-trained layer called %s" % name)
            return None
        self.para_dict[name] = [weights, biases]
        self.weight_decay += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)

        # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
        fc = tf.matmul(bottom, weights) + biases
        return fc


# %load extractor.py
import numpy as np
import tensorflow as tf

class Extractor(Model):
    def __init__(self, shape=(224, 224, 3), scope_name="VGG16"):
        super().__init__()
        # input information
        self.H, self.W, self.C = shape
        self.scope_name = scope_name

        # parameter dictionary
        self.para_dict = dict()
        
    def build(self, vgg16_npy_path):
        """
        load pre-trained weights from path
        :param vgg16_npy_path: file path of vgg16 pre-trained weights
        """

        # input placeholder
        self.x = tf.placeholder(tf.float32, [None, self.H, self.W, self.C])
        self.is_train = tf.placeholder(tf.bool)
        
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.x)
        self.x = tf.concat(axis=3,
                           values=[blue - 103.939,
                                   green - 116.779,
                                   red - 123.68,])
        
        assert self.x.get_shape().as_list()[1:] == [self.H, self.W, self.C]

        # load pre-trained weights
        if isinstance(vgg16_npy_path,dict):
            self.data_dict = vgg16_npy_path
            print("parameters loaded")
        else:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            print("npy file loaded")

        ### pre-trained VGG-16 start ###
        conv1_1 = self.conv_layer( self.x, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool_layer(conv1_2, "pool1")

        conv2_1 = self.conv_layer(  pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool_layer(conv2_2, "pool2")

        conv3_1 = self.conv_layer(  pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool_layer(conv3_3, "pool3")

        conv4_1 = self.conv_layer(  pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4   = self.max_pool_layer(conv4_3, "pool4")

        conv5_1 = self.conv_layer(  pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool_layer(conv5_3, "pool5")
        # flatten = tf.reduce_mean(conv5_3, [1,2])
        ### pre-trained VGG-16 end ###
        
        flatten = self.flatten_layer(pool5)
        self.output = flatten

class MQ1(Model):
    def __init__(self, scope_name=""):
        super().__init__()
        self.scope_name = scope_name
        
    def build(self, input_dim, output_dim):
        with tf.variable_scope(name_or_scope=self.scope_name) as scope:
            self.x = tf.placeholder(tf.float32, shape=[None, input_dim], name="input")
            self.y = tf.placeholder(tf.int32, shape=[None, output_dim], name="output")
            self.is_train = tf.placeholder(tf.bool)
        
            fc1 = self.dense_layer(bottom=self.x, name="fc1", shape=[input_dim, 2048])
            fc1 = tf.contrib.layers.batch_norm(fc1, is_training=self.is_train, scope="bn1", decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
            fc1 = tf.nn.relu(fc1)
            
            fc2 = self.dense_layer(bottom=fc1, name="fc2", shape=[2048, 768])
            fc2 = tf.contrib.layers.batch_norm(fc2, is_training=self.is_train, scope="bn2", decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
            fc2 = tf.nn.relu(fc2)
            
            self.logits = self.dense_layer(bottom=fc2, name="logits", shape=[768, output_dim])
            self.pred = tf.argmax(self.logits, axis=1, name="pred")
            
            def train_operation():
                self.true = tf.argmax(self.y, axis=1, name="true")
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits, name='loss'))
            
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(x=self.pred, y=self.true), tf.float32))
                return True

            _ = tf.cond(self.is_train, train_operation, lambda: False)

class MQ2(Model):
    def __init__(self, scope_name=""):
        super().__init__()
        self.scope_name = scope_name
        
    def build(self, lstm_units, max_seq_len, input_dim, output_dim):
        with tf.variable_scope(name_or_scope=self.scope_name) as scope:
            self.x = tf.placeholder(tf.float32, shape=[None, max_seq_len, input_dim], name='input')
            self.y = tf.placeholder(tf.int32, shape=[None, output_dim], name='output')
            self.seq_end_index = tf.placeholder(tf.int32, shape=[None], name='seq_end_index')
            self.is_train = tf.placeholder(tf.bool)
            
            # get batch size
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.max_seq_len = max_seq_len
            self.lstm_units = lstm_units
            self.batch_size = tf.shape(self.x)[0]

            # stacked rnn cell
            cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n, forget_bias=1.0) for n in lstm_units]

            stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
      
            # dynamic_rnn
            step_rnn_output, step_rnn_state = tf.nn.dynamic_rnn(stacked_rnn_cell,
                                                                self.x,
                                                                sequence_length=self.seq_end_index,
                                                                dtype=tf.float32)
            # shape meaning: batch, step, feature
            print(step_rnn_output.get_shape().as_list())
            
            # make output in shape, step x batch, feature 
            step_rnn_output = tf.reshape(step_rnn_output, [-1, self.lstm_units[1]])
            print(step_rnn_output.get_shape().as_list())
            
            # select the output at step*50*i + seq_end_index[i]
            output_index = tf.range(0, self.batch_size)*self.max_seq_len + self.seq_end_index - 1
            
            #for i in seq_end_index
            self.rnn_output = tf.gather(step_rnn_output, output_index)
            print(self.rnn_output.get_shape().as_list())
            
            self.rnn_output = self.dropout_layer(bottom=self.rnn_output, keep_prob=0.8, is_train=self.is_train)
            
            fc1 = self.dense_layer(bottom=self.rnn_output, name="fc1", shape=[self.lstm_units[-1], 512])
            fc1 = tf.contrib.layers.batch_norm(fc1, is_training=self.is_train, scope="bn1", decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
            fc1 = tf.nn.relu(fc1)
            
            self.logits = self.dense_layer(bottom=fc1, name='logits', shape=[512, self.output_dim])
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits, name='loss'))
            
            
            self.pred = tf.argmax(self.logits, axis=1, name="pred")
            self.true = tf.argmax(self.y, axis=1, name="true")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(x=self.pred, y=self.true), tf.float32))