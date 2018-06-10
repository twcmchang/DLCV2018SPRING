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
from model import MQ2
from utils import load_frame, one_hot_encoding, pad_feature_maxlen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_pkl_file', type=str, default='valid_codes_full.pkl', help='directory to store checkpointed models')
    parser.add_argument('--valid_video_list', type=str, default='HW5_data/TrimmedVideos/label/gt_valid.csv', help='directory to store checkpointed models')   
    parser.add_argument('--save_dir', type=str, default='save_Q2/', help='directory of saveing model.ckpt')
    parser.add_argument('--run_tsne', type=bool, default=False, help='run tSNE to visualize learned features or not')

    FLAG = parser.parse_args()

    print("===== create directory =====")
    if not os.path.exists(FLAG.save_dir):
        print("please specify the model directory.")

    test(FLAG)

def test(FLAG):
    output_dim = 11
    valid_list = getVideoList(FLAG.valid_video_list)

    dvalid = pd.DataFrame.from_dict(valid_list)

    xtest  = load_frame(FLAG.valid_pkl_file)

    # model
    scope_name = "M2"
    model = MQ2(scope_name=scope_name)
    model.build(lstm_units=[1024, 1024], max_seq_len=25, input_dim= 40960, output_dim=output_dim)

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


        Xtest, Xtest_end_index = pad_feature_maxlen(xtest, max_len=model.max_seq_len)
        pred, rnn_output = sess.run([model.pred, model.rnn_output],
                            feed_dict={model.x: Xtest,
                                    model.seq_end_index: Xtest_end_index,
                                    model.is_train:False})

        np.savetxt(X=pred.astype(int), fname='p2_result.txt',fmt='%s')
        print("save p2_result.txt")

        if FLAG.run_tsne:
            from sklearn.manifold import TSNE
            rnn_tsne = TSNE(n_components=2, perplexity=30.0, random_state=5566).fit_transform(rnn_output)
            
            labels = np.array(dvalid.Action_labels).astype('int32')
            plt.figure(0)
            for i in range(output_dim):
                xplot = rnn_tsne[np.where(labels==i)[0]]
                plt.scatter(xplot[:,0], xplot[:,1], label=i)
            plt.legend()
            plt.title("RNN-based features")
            plt.xlabel("tSNE-1")
            plt.ylabel("tSNE-2")
            plt.tight_layout()
            plt.show()
            plt.savefig('rnn_tsne.png')
            print("save rnn_tsne.png")
