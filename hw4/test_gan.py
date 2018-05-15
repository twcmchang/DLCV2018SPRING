import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import skimage.transform
import imageio
from gan import GAN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_dir', type=str, default='plot', help='directory to store figure')
    parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')
    FLAG = parser.parse_args()

    print("===== create directory =====")
    if not os.path.exists(FLAG.plot_dir):
        os.makedirs(FLAG.plot_dir)
        print("makedir %s"%FLAG.plot_dir)

    plot_curve(FLAG)
    inference(FLAG)

def plot_curve(FLAG):
    filepath = os.path.join(FLAG.save_dir, "history_dict.npy")
    if not os.path.exists(filepath):
        print("Not exist history_dict.npy in %s" % FLAG.save_dir)
    else:
        d = np.load(filepath, encoding='latin1').item()
        a= pd.DataFrame.from_dict(d)

        fig, ax = plt.subplots(1,2, figsize=(16,6))
        ax[0].plot(range(a.shape[0]), a['G_loss'])
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('loss')
        ax[0].set_title('Generator')
        ax[0].legend(loc="right", fontsize=14)
        ax[1].plot(range(a.shape[0]), a['D_fake_accu'])
        ax[1].plot(range(a.shape[0]), a['D_real_accu'])
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('accuracy')
        ax[1].set_title('Discriminator')
        ax[1].legend(loc="right", fontsize=14)
        plt.savefig(os.path.join(FLAG.plot_dir, 'fig2_2.jpg'))
        plt.close(fig)

def inference(FLAG):
    FLAG_save_dir = FLAG.save_dir
    FLAG_plot_dir = FLAG.plot_dir
    FLAG_batch_size = 32
    FLAG_n_dim = 100

    gan = GAN()
    gan.build(n_dim=FLAG_n_dim, shape=(64, 64, 3))

    def initialize_uninitialized(sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars): 
            sess.run(tf.variables_initializer(not_initialized_vars))

    def res_plot(samples, n_row, n_col):     
        fig = plt.figure(figsize=(n_col*2, n_row*2))
        gs = gridspec.GridSpec(n_row, n_col)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(64, 64, 3))
        return fig

    with tf.Session() as sess:
        if FLAG_save_dir is not None:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAG_save_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored %s" % ckpt.model_checkpoint_path)
                sess.run(tf.global_variables())
            else:
                print("No model checkpoint in %s" % FLAG_save_dir)
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.global_variables())
        print("Initialized")
        print("Plot saved in %s" % FLAG_plot_dir)
        
        # hyper parameters
        batch_size = FLAG_batch_size

        # re-initialize
        initialize_uninitialized(sess)
                
        # plot
        np.random.seed(296)
        Xplot = sess.run(gan.G_image,
                feed_dict={gan.random_sample: np.random.uniform(-1, 1, [batch_size, gan.n_dim]).astype(np.float32),
                        gan.is_train: False})
        fig = res_plot(Xplot, int(batch_size/8), 8)
        plt.savefig(os.path.join(FLAG_plot_dir, 'fig2_3.jpg'), bbox_inches='tight')
        plt.close(fig)
        

if __name__ == '__main__':
    main()         