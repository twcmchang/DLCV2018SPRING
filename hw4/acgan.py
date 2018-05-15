import os
import time
import numpy as np
import tensorflow as tf
from gan import GAN

class ACGAN(GAN):
    def __init_(self):
        super().__init__()
        
    def build(self, aux_dim=1, n_dim=512, shape=(64,64,3)):
        """
        load pre-trained weights from path
        :param vgg16_npy_path: file path of vgg16 pre-trained weights
        """
        # input information
        self.H, self.W, self.C = shape
        self.n_dim = n_dim
        self.aux_dim = aux_dim
        
        # parameter dictionary
        self.para_dict = dict()
        self.data_dict = dict()
        self.net_shape = dict()

        # input placeholder
        self.x = tf.placeholder(tf.float32, [None, self.H, self.W, self.C])
        self.random_sample = tf.placeholder(tf.float32, [None, self.n_dim])
        self.aux_labels = tf.placeholder(tf.float32, shape=[None, self.aux_dim])
        self.is_train = tf.placeholder(tf.bool)
        
        # normalize inputs
        assert self.x.get_shape().as_list()[1:] == [self.H, self.W, self.C]
        

        with tf.variable_scope("Generator",reuse=tf.AUTO_REUSE):
            self.G_image = self.generator(self.random_sample, self.aux_labels)
            
        with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
            self.D_real = self.dense_layer(self.discriminator(self.x), n_hidden=1, name='D_output')
            self.D_fake = self.dense_layer(self.discriminator(self.G_image), n_hidden=1, name='D_output')
            
            self.D_aux_real = self.dense_layer(self.discriminator(self.x), n_hidden=self.aux_dim, name='aux_output')
            self.D_aux_fake = self.dense_layer(self.discriminator(self.G_image), n_hidden=self.aux_dim, name='aux_output')
        
        real_equality = tf.equal(tf.cast(tf.sigmoid(self.D_real) > 0.5, tf.float32), tf.ones(shape=tf.shape(self.D_real)))
        self.D_real_accu = tf.reduce_mean(tf.cast(real_equality, tf.float32))
        
        fake_equality = tf.equal(tf.cast(tf.sigmoid(self.D_fake) > 0.5, tf.float32), tf.zeros(shape=tf.shape(self.D_fake)))
        self.D_fake_accu = tf.reduce_mean(tf.cast(fake_equality, tf.float32))
        
        # loss of discriminator
        self.D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real,
                                                                                  labels=tf.ones(shape=tf.shape(self.D_real))))
        self.D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake,
                                                                                  labels=tf.zeros(shape=tf.shape(self.D_fake))))
        self.D_loss = self.D_real_loss + self.D_fake_loss
        
        # loss of generator
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake,labels=tf.ones(shape=tf.shape(self.D_real))))
        

        self.D_aux_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_aux_real,
                                                                                    labels=self.aux_labels))
        self.D_aux_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_aux_fake,
                                                                                    labels=self.aux_labels))
        self.D_aux_total_loss = self.D_aux_real_loss + self.D_aux_fake_loss
        
        self.D_loss = self.D_loss + self.D_aux_total_loss
        self.G_loss = self.G_loss + self.D_aux_total_loss
        print("GAN graph built.")
        
    def discriminator(self, input_image):
        conv1 = self.conv_bn_layer(input_image, shape=(4,4,3,32), stride=2, name="conv1")
        conv2 = self.conv_bn_layer(conv1 , shape=(4,4,32,64), stride=2, name="conv2")
        conv3 = self.conv_bn_layer(conv2 , shape=(4,4,64,128), stride=2, name="conv3")
        conv4 = self.conv_bn_layer(conv3 , shape=(4,4,128,256), stride=2, name="conv4")
        flatten = self.flatten_layer(conv4, name='flatten')
        #output = self.dense_layer(flatten, n_hidden=1, name='D_output')
        return flatten
    
    def generator(self, sample_input, sample_aux_input):
        sample_input = tf.concat([sample_input, sample_aux_input], axis=1)
        deconv_fc1 = self.dense_layer(sample_input, n_hidden=4096, name='deconv_fc1')
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