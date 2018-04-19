import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from model import VGG16
from utils import read_images, read_masks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_from', type=str, default='vgg16.npy', help='pre-trained weights')
    parser.add_argument('--save_dir', type=str, default=None, help='directory to store checkpointed models')
    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='dataset in use')
    parser.add_argument('--prof_type', type=str, default='all-one', help='type of profile coefficient')
    parser.add_argument('--output', type=str, default='output.csv', help='output filename (csv)')
    parser.add_argument('--keep_prob', type=float, default=1.0, help='dropout keep probability for fc layer')
    parser.add_argument('--fidelity', type=float, default=None, help='fidelity in use') 

    FLAG = parser.parse_args()
    test(FLAG)

def test(FLAG):
    print("Reading dataset...")
    if FLAG.dataset == 'CIFAR-10':
        test_data  = CIFAR10(train=False)
        vgg16 = VGG16(classes=10)
    elif FLAG.dataset == 'CIFAR-100':
        test_data  = CIFAR100(train=False)
        vgg16 = VGG16(classes=100)
    else:
        raise ValueError("dataset should be either CIFAR-10 or CIFAR-100.")

    Xtest, Ytest = test_data.test_data, test_data.test_labels

    if FLAG.fidelity is not None:
        data_dict = np.load(FLAG.init_from, encoding='latin1').item()
        data_dict = dpSparsifyVGG16(data_dict,FLAG.fidelity)
        vgg16.build(vgg16_npy_path=data_dict, prof_type=FLAG.prof_type, conv_pre_training=True, fc_pre_training=True)
        print("Build model from %s using dp=%s" % (FLAG.init_from, str(FLAG.fidelity*100)))
    else:
        vgg16.build(vgg16_npy_path=FLAG.init_from, prof_type=FLAG.prof_type, conv_pre_training=True, fc_pre_training=True)
        print("Build full model from %s" % (FLAG.init_from))

    with tf.Session() as sess:
        if FLAG.save_dir is not None:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAG.save_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, checkpoint)
                print("Model restored %s" % checkpoint)
                sess.run(tf.global_variables())
            else:
                print("No model checkpoint in %s" % FLAG.save_dir)
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.global_variables())
        print("Initialized")
        output = []
        for dp_i in dp:
            accu = sess.run(vgg16.accu_dict[str(int(dp_i*100))], feed_dict={vgg16.x: Xtest[:5000,:], vgg16.y: Ytest[:5000,:], vgg16.is_train: False})
            accu2 = sess.run(vgg16.accu_dict[str(int(dp_i*100))], feed_dict={vgg16.x: Xtest[5000:,:], vgg16.y: Ytest[5000:,:], vgg16.is_train: False})
            output.append((accu+accu2)/2)
            print("At DP={dp:.4f}, accu={perf:.4f}".format(dp=dp_i, perf=(accu+accu2)/2))
        res = pd.DataFrame.from_dict({'DP':[int(dp_i*100) for dp_i in dp],'accu':output})
        res.to_csv(FLAG.output, index=False)
        print("Write into %s" % FLAG.output)


if __name__ == '__main__':
	main()
