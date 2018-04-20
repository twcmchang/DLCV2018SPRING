import os
import time
import argparse
import numpy as np
import tensorflow as tf

import skimage.transform
import imageio

from model import VGG16
from utils import read_images, read_masks, read_list, label2rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='hw3-train-validation/validation/', help='testing file directory')
    parser.add_argument('--init_from', type=str, default='keras-vgg16.npy', help='pre-trained weights')
    parser.add_argument('--mode', type=str, default='FCN32s', help='FCN mode: FCN32s, FCN16s, FCN8s')
    parser.add_argument('--save_dir', type=str, default=None, help='directory to store checkpointed models')
    parser.add_argument('--plot_dir', type=str, default='pred', help='dataset in use')

    FLAG = parser.parse_args()

    print("===== create directory =====")
    if not os.path.exists(FLAG.plot_dir):
        os.makedirs(FLAG.plot_dir)
    test(FLAG)

def test(FLAG):
    print("Reading dataset...")
    # load data
    file_list = [FLAG.test_dir+file.replace('_sat.jpg','') for file in os.listdir(FLAG.test_dir) if file.endswith('_sat.jpg')]
    file_list.sort()
    Xtest, Ytest = read_list(file_list)

    vgg16 = VGG16(classes=7, shape=(256,256,3))
    vgg16.build(vgg16_npy_path=FLAG.init_from, mode=FLAG.mode)

    def initialize_uninitialized(sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars): 
                sess.run(tf.variables_initializer(not_initialized_vars))

    with tf.Session() as sess:
        if FLAG.save_dir is not None:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAG.save_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored %s" % ckpt.model_checkpoint_path)
                sess.run(tf.global_variables())
            else:
                print("No model checkpoint in %s" % FLAG.save_dir)
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.global_variables())
        print("Initialized")

        print("Plot saved in %s" % FLAG.plot_dir)
        for i, fname in enumerate(file_list):
            Xplot = sess.run(vgg16.pred,feed_dict={vgg16.x: Xtest[i:(i+1),:],
                                        vgg16.y: Ytest[i:(i+1),:],
                                        vgg16.is_train: False})
            saveimg = skimage.transform.resize(Xplot[0],output_shape=(512,512),order=0,preserve_range=True,clip=False)
            saveimg = label2rgb(saveimg)
            imageio.imwrite(os.path.join(FLAG.plot_dir,os.path.basename(fname)+"_mask.png"), saveimg)
            print(os.path.join(FLAG.plot_dir,os.path.basename(fname)+"_mask.png"))

if __name__ == '__main__':
	main()
