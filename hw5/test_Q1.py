import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle

from reader import getVideoList
from model import MQ1
from utils import load_aggregate_frame, one_hot_encoding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_pkl_file', type=str, default='valid_codes_full.pkl', help='directory to store checkpointed models')
    parser.add_argument('--valid_video_list', type=str, default='HW5_data/TrimmedVideos/label/gt_valid.csv', help='directory to store checkpointed models')   
    parser.add_argument('--save_dir', type=str, default='save_Q1/', help='directory of saveing model.ckpt')
    parser.add_argument('--run_tsne', type=bool, default=False, help='run tSNE to visualize learned features or not')

    FLAG = parser.parse_args()

    print("===== create directory =====")
    if not os.path.exists(FLAG.save_dir):
        print("please specify the model directory.")

    test(FLAG)

def test(FLAG):

    valid_list = getVideoList(FLAG.valid_video_list)

    dvalid = pd.DataFrame.from_dict(valid_list)

    Xtest  = load_aggregate_frame(FLAG.valid_pkl_file)
    Ytest  = np.array(dvalid.Action_labels).astype('int32')

    scope_name = "Q1"
    model = MQ1(scope_name=scope_name)
    model.build(input_dim=Xtest.shape[1], output_dim=Ytest.shape[1])

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

        pred, cnn_output = sess.run([model.pred, model.output],
                            feed_dict={model.x: Xtest,
                                    model.is_train:False})
        
        print("save p1_valid.txt")
        np.savetxt(X=pred.astype(int), fname='p1_valid.txt', fmt='%s')
        
        if FLAG.run_tsne:
            from sklearn.manifold import TSNE
            cnn_tsne = TSNE(n_components=2, perplexity=0.0, random_state=5566).fit_transform(cnn_output)
            
            print("save cnn_tsne.png")
            labels = np.array(dvalid.Action_labels).astype('int32')
            plt.figure(0)
            for i in range(Xtest.shape[1]):
                xplot = cnn_tsne[np.where(labels==i)[0]]
                plt.scatter(xplot[:,0], xplot[:,1], label=i)
            plt.legend()
            plt.title("CNN-based features")
            plt.xlabel("tSNE-1")
            plt.ylabel("tSNE-2")
            plt.tight_layout()
            plt.show()
            plt.savefig('cnn_tsne.png')