import os
import time
import argparse
import numpy as np
import tensorflow as tf

from progress.bar import Bar
from ipywidgets import IntProgress
from IPython.display import display
import skimage.transform
import imageio

from model import VAE
from utils import read_dataset

TRAIN_CSV = "hw4_data/train.csv"
TRAIN_DIR = "hw4_data/train/"
TEST_CSV = "hw4_data/test.csv"
TEST_DIR = "hw4_data/test/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')
    parser.add_argument('--lr', type=float, default=1e-4, help='starting learning rate')
    parser.add_argument('--n_dim', type=int, default=512, help='dimension of hidden code')
    parser.add_argument('--lambda_KL', type=float, default=1e-5, help='weight of KL divergence loss')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--note', type=str, default='', help='argument for taking notes')

    FLAG = parser.parse_args()

    print("===== create directory =====")
    if not os.path.exists(FLAG.save_dir):
        os.makedirs(FLAG.save_dir)

    train(FLAG)

def train(FLAG):
    print("Reading dataset...")
    # load data
    Xtrain, df_train = read_dataset(TRAIN_CSV, TRAIN_DIR)
    Xtest , df_test  = read_dataset(TEST_CSV , TEST_DIR)

    vae = VAE()
    vae.build(lambda_KL=FLAG.lambda_KL,n_dim=FLAG.n_dim, batch_size=FLAG.batch_size, shape=Xtrain.shape[1:])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    checkpoint_path = os.path.join(FLAG.save_dir, 'model.ckpt')

    def initialize_uninitialized(sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars): 
                sess.run(tf.variables_initializer(not_initialized_vars))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # hyper parameters
        batch_size = FLAG.batch_size
        epoch = 500
        early_stop_patience = 50
        min_delta = 0.0001
        opt_type = 'adam'

        # recorder
        epoch_counter = 0

        # optimizer
        global_step = tf.Variable(0, trainable=False)

        # Passing global_step to minimize() will increment it at each step.
        if opt_type is 'sgd':
            start_learning_rate = FLAG.lr
            half_cycle = 2000
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, half_cycle, 0.5, staircase=True)
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        else:
            start_learning_rate = FLAG.lr
            half_cycle = 2000
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, half_cycle, 0.5, staircase=True)
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        obj = vae.train_op
        train_op = opt.minimize(obj, global_step=global_step)

        # progress bar
        ptrain = IntProgress()
        pval = IntProgress()
        display(ptrain)
        display(pval)
        ptrain.max = int(Xtrain.shape[0]/batch_size)
        pval.max = int(Xtest.shape[0]/batch_size)

        # re-initialize
        initialize_uninitialized(sess)

        # reset due to adding a new task
        patience_counter = 0
        current_best_val_loss = np.float('Inf')

        # optimize when the aggregated obj
        while(patience_counter < early_stop_patience and epoch_counter < epoch):

            # start training
            stime = time.time()
            bar_train = Bar('Training', max=int(Xtrain.shape[0]/batch_size), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
            bar_val =  Bar('Validation', max=int(Xtest.shape[0]/batch_size), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
            
            train_loss = 0.0
            train_reconstruction_loss = 0.0
            train_kl_loss = 0.0
            for i in range(int(Xtrain.shape[0]/batch_size)):
                st = i*batch_size
                ed = (i+1)*batch_size
                loss, reconstruction_loss, kl_loss ,_ = sess.run([obj, vae.loss['reconstruction'], vae.loss['KL_loss'], train_op],
                                    feed_dict={vae.x: Xtrain[st:ed,:],
                                                vae.y: Xtrain[st:ed,:],
                                                vae.is_train: True})
                train_loss += loss
                train_reconstruction_loss += reconstruction_loss
                train_kl_loss += kl_loss
                ptrain.value +=1
                ptrain.description = "Training %s/%s" % (ptrain.value, ptrain.max)
            train_loss = train_loss/ptrain.value
            train_reconstruction_loss = train_reconstruction_loss/ptrain.value
            train_kl_loss = train_kl_loss/ptrain.value

            # validation
            val_loss = 0
            val_reconstruction_loss = 0.0
            val_kl_loss = 0.0
            for i in range(int(Xtest.shape[0]/batch_size)):
                st = i*batch_size
                ed = (i+1)*batch_size
                loss, reconstruction_loss, kl_loss = sess.run([obj, vae.loss['reconstruction'], vae.loss['KL_loss']],
                                    feed_dict={vae.x: Xtest[st:ed,:],
                                                vae.y: Xtest[st:ed,:],
                                                vae.is_train: False})
                val_loss += loss
                val_reconstruction_loss += reconstruction_loss
                val_kl_loss += kl_loss
                pval.value += 1
                pval.description = "Testing %s/%s" % (pval.value, pval.value)
            val_loss = val_loss/pval.value
            val_reconstruction_loss = val_reconstruction_loss/pval.value
            val_kl_loss = val_kl_loss/pval.value
            
            # plot
            # if epoch_counter%10 == 0:
            #     Xplot = sess.run(vae.output,
            #             feed_dict={vae.x: Xtest[:,:],
            #                         vae.y: Xtest[:,:],
            #                         vae.is_train: False})
            #     for i, fname in enumerate(track):
            #         imageio.imwrite(os.path.join(FLAG.save_dir,os.path.basename(fname)+"_pred_"+str(epoch_counter)+".png"), saveimg)
            #         print(os.path.join(FLAG.save_dir,os.path.basename(fname)+"_pred_"+str(epoch_counter)+".png"))
                
            # early stopping check
            if (current_best_val_loss - val_loss) > min_delta:
                current_best_val_loss = val_loss
                patience_counter = 0
                saver.save(sess, checkpoint_path, global_step=epoch_counter)
                print("save in %s" % checkpoint_path)
            else:
                patience_counter += 1

            # shuffle Xtrain and Ytrain in the next epoch
            idx = np.random.permutation(Xtrain.shape[0])
            Xtrain= Xtrain[idx,:,:,:]

            # epoch end
            epoch_counter += 1

            ptrain.value = 0
            pval.value = 0
            bar_train.finish()
            bar_val.finish()

            print("Epoch %s (%s), %s sec >> train loss: %.4f, train recon loss: %.4f, train kl loss: %.4f, val loss: %.4f, val recon loss: %.4f, val kl loss: %.4f" % (epoch_counter, patience_counter, round(time.time()-stime,2), train_loss, train_reconstruction_loss, train_kl_loss, val_loss, val_reconstruction_loss ,val_kl_loss))
        
        # para_dict = sess.run(vgg16.para_dict)
        # np.save(os.path.join(FLAG.save_dir, "para_dict.npy"), para_dict)
        # print("save in %s" % os.path.join(FLAG.save_dir, "para_dict.npy"))

        FLAG.optimizer = opt_type
        FLAG.lr = start_learning_rate
        FLAG.batch_size = batch_size
        FLAG.epoch_end = epoch_counter
        FLAG.val_loss = current_best_val_loss

        header = ''
        row = ''
        for key in sorted(vars(FLAG)):
            if header is '':
                header = key
                row = str(getattr(FLAG, key))
            else:
                header += ","+key
                row += ","+str(getattr(FLAG,key))
        row += "\n"
        if os.path.exists("/home/cmchang/DLCV2018SPRING/hw4/model.csv"):
            with open("/home/cmchang/DLCV2018SPRING/hw4/model.csv", "a") as myfile:
                myfile.write(row)
        else:
            with open("/home/cmchang/DLCV2018SPRING/hw4/model.csv", "w") as myfile:
                myfile.write(header)
                myfile.write(row)

if __name__ == '__main__':
    main()
