import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
matplotlib.use('agg')
import pickle

from reader import getVideoList
from model import MQ1
from utils import load_aggregate_frame, one_hot_encoding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pkl_file', type=str, default='train_codes_full.pkl', help='directory to store checkpointed models')
    parser.add_argument('--valid_pkl_file', type=str, default='valid_codes_full.pkl', help='directory to store checkpointed models')
    parser.add_argument('--train_video_list', type=str, default='HW5_data/TrimmedVideos/label/gt_train.csv', help='directory to store checkpointed models')
    parser.add_argument('--valid_video_list', type=str, default='HW5_data/TrimmedVideos/label/gt_valid.csv', help='directory to store checkpointed models')   
    parser.add_argument('--save_dir', type=str, default='save_Q1/', help='argument for taking notes')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')

    FLAG = parser.parse_args()

    print("===== create directory =====")
    if not os.path.exists(FLAG.save_dir):
        os.makedirs(FLAG.save_dir)

    train(FLAG)

def train(FLAG):

    train_list = getVideoList(FLAG.train_video_list)
    valid_list = getVideoList(FLAG.valid_video_list)

    dtrain = pd.DataFrame.from_dict(train_list)
    dvalid = pd.DataFrame.from_dict(valid_list)

    Xtrain = load_aggregate_frame("train_codes_full.pkl")
    Xtest  = load_aggregate_frame("valid_codes_full.pkl")

    Ytrain = np.array(dtrain.Action_labels).astype('int32')
    Ytest  = np.array(dvalid.Action_labels).astype('int32')
    Ytrain = one_hot_encoding(Ytrain, 11)
    Ytest = one_hot_encoding(Ytest, 11)

    scope_name = "Q1"
    model = MQ1(scope_name=scope_name)
    model.build(input_dim=Xtrain.shape[1], output_dim=Ytrain.shape[1])

    train_vars = list()
    for var in tf.trainable_variables():
        if model.scope_name in var.name:
            train_vars.append(var)
            
    # optimizer
    learning_rate = FLAG.lr
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(model.loss, var_list=train_vars)

    def initialize_uninitialized(sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars): 
                sess.run(tf.variables_initializer(not_initialized_vars))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # hyper parameters
        batch_size = 32
        epoch = 30
        early_stop_patience = 5
        min_delta = 0.0001

        # recorder
        epoch_counter = 0
        history = list()

        # re-initialize
        initialize_uninitialized(sess)

        # reset due to adding a new task
        patience_counter = 0
        current_best_val_accu = 0

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        checkpoint_path = os.path.join(FLAG.save_dir, 'model.ckpt')

        # optimize when the aggregated obj
        while(patience_counter < early_stop_patience and epoch_counter < epoch):

            # start training
            stime = time.time()

            train_loss, train_accu = 0.0, 0.0    
            for i in range(int(Xtrain.shape[0]/batch_size)):
                st = i*batch_size
                ed = (i+1)*batch_size
                loss, accu , _, logits = sess.run([model.loss, model.accuracy, model.logits, train_op],
                                    feed_dict={model.x: Xtrain[st:ed],
                                                model.y: Ytrain[st:ed],
                                            model.is_train:True})
                train_loss += loss
                train_accu += accu
            train_loss = train_loss/(Xtrain.shape[0]/batch_size)
            train_accu = train_accu/(Xtrain.shape[0]/batch_size)
            
            val_loss, val_accu = 0.0, 0.0
            for i in range(int(Xtest.shape[0]/batch_size)):
                st = i*batch_size
                ed = (i+1)*batch_size
                loss, accu , logits = sess.run([model.loss, model.accuracy, model.logits],
                                    feed_dict={model.x: Xtest[st:ed],
                                                model.y: Ytest[st:ed],
                                            model.is_train:False})
                val_loss += loss
                val_accu += accu
            val_loss = val_loss/(Xtest.shape[0]/batch_size)
            val_accu = val_accu/(Xtest.shape[0]/batch_size)
            
            print("Epoch %s (%s), %s sec >> train loss: %.4f, train accu: %.4f, val loss: %.4f, val accu: %.4f" % (epoch_counter, patience_counter, round(time.time()-stime,2), train_loss, train_accu, val_loss, val_accu))
            history.append([train_loss, train_accu, val_loss, val_accu])
            
            # early stopping check
            if (val_accu - current_best_val_accu) > min_delta:
                current_best_val_accu = val_accu
                patience_counter = 0
                saver.save(sess, checkpoint_path, global_step=epoch_counter)
                print("save in %s" % checkpoint_path)
                para_dict = sess.run(model.para_dict)
                np.save(os.path.join(FLAG.save_dir, "para_dict.npy"), para_dict)
                print("save in %s" % os.path.join(FLAG.save_dir, "para_dict.npy"))
            else:
                patience_counter += 1

            # shuffle Xtrain and Ytrain in the next epoch
            idx = np.random.permutation(Xtrain.shape[0])
            Xtrain, Ytrain = Xtrain[idx,:], Ytrain[idx,:]

            # epoch end
            epoch_counter += 1
        # end of training
    # end of session
    df = pd.DataFrame(history)
    df.columns = ['train_loss', 'train_accu', 'val_loss', 'val_accu']
    plt.figure(0)
    df[['train_loss', 'val_loss']].plot()
    plt.savefig(os.path.join(FLAG.save_dir, 'loss.png'))
    plt.close()

    plt.figure(0)
    df[['train_accu', 'val_accu']].plot()
    plt.savefig(os.path.join(FLAG.save_dir, 'accu.png'))
    plt.close()
