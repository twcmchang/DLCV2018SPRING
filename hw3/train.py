import os
import time
import argparse
import numpy as np
import tensorflow as tf

from progress.bar import Bar
from ipywidgets import IntProgress
from IPython.display import display

from model import VGG16
from utils import read_images, read_masks

# import imgaug as ia
# from imgaug import augmenters as iaa
# sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# transform = iaa.Sequential([
#     sometimes(iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)})),
#     sometimes(iaa.Affine(scale={"x": (0.85, 1.15), "y":(0.85, 1.15)})),
#     sometimes(iaa.Affine(rotate=(-45, 45))),
#     sometimes(iaa.Add((-10,10), per_channel=0.5)),
#     sometimes(iaa.Fliplr(0.5))
# ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_from', type=str, default='keras-vgg16.npy', help='pre-trained weights')
    parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')
    parser.add_argument('--decay', type=float, default=0.0, help='multiplier for weight decay')
    parser.add_argument('--keep_prob', type=float, default=1.0, help='dropout keep probability for fc layer')    
    parser.add_argument('--note', type=str, default='', help='argument for taking notes')

    FLAG = parser.parse_args()

    print("===== create directory =====")
    if not os.path.exists(FLAG.save_dir):
        os.makedirs(FLAG.save_dir)

    train(FLAG)
    

def train(FLAG):
    print("Reading dataset...")
    vgg16 = VGG16(classes=1, shape=(512,512,3))
    vgg16.build(vgg16_npy_path="keras-vgg16.npy")

    # load data
    Xtrain, Ytrain = read_images("hw3-train-validation/train/"), read_masks("hw3-train-validation/train/") 
    Xtest, Ytest = read_images("hw3-train-validation/validation/"), read_masks("hw3-train-validation/validation/"),  

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    checkpoint_path = os.path.join(FLAG.save_dir, 'model.ckpt')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

       # hyper parameters
        batch_size = 64
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
            start_learning_rate = 1e-4
            half_cycle = 20000
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, half_cycle, 0.5, staircase=True)
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        else:
            start_learning_rate = 1e-4
            half_cycle = 20000
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, half_cycle, 0.5, staircase=True)
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        obj = vgg16.loss
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
        current_best_val_accu = 0

        # optimize when the aggregated obj
        while(patience_counter < early_stop_patience and epoch_counter < epoch):
            
            def load_batches():
                for i in range(int(Xtrain.shape[0]/batch_size)):
                    st = i*batch_size
                    ed = (i+1)*batch_size
                    batch = ia.Batch(images=Xtrain[st:ed,:,:,:], data=Ytrain[st:ed,:])
                    yield batch

            batch_loader = ia.BatchLoader(load_batches)
            bg_augmenter = ia.BackgroundAugmenter(batch_loader=batch_loader, augseq=transform, nb_workers=4)

            # start training
            stime = time.time()
            bar_train = Bar('Training', max=int(Xtrain.shape[0]/batch_size), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
            bar_val =  Bar('Validation', max=int(Xtest.shape[0]/batch_size), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
            train_loss, train_accu = 0.0, 0.0
            while True:
                batch = bg_augmenter.get_batch()
                if batch is None:
                    print("Finished epoch.")
                    break
                x_images_aug = batch.images_aug
                y_images = batch.data
                loss, accu, _ = sess.run([obj, vgg16.accuracy, train_op], 
                                        feed_dict={vgg16.x: x_images_aug,
                                        vgg16.y: y_images,
                                        vgg16.is_train: True})
                bar_train.next()
                train_loss += loss
                train_accu += accu
                ptrain.value +=1
                ptrain.description = "Training %s/%s" % (ptrain.value, ptrain.max)
            train_loss = train_loss/ptrain.value
            train_accu = train_accu/ptrain.value
            batch_loader.terminate()
            bg_augmenter.terminate()

            # validation
            val_loss = 0
            val_accu = 0
            for i in range(int(Xtest.shape[0]/200)):
                st = i*200
                ed = (i+1)*200
                loss, accu = sess.run([obj, vgg16.accuracy],
                                    feed_dict={vgg16.x: Xtest[st:ed,:],
                                                vgg16.y: Ytest[st:ed,:],
                                                vgg16.is_train: False})
                val_loss += loss
                val_accu += accu
                pval.value += 1
                pval.description = "Testing %s/%s" % (pval.value, pval.value)
            val_loss = val_loss/pval.value
            val_accu = val_accu/pval.value

            # early stopping check
            if (val_accu - current_best_val_accu) > min_delta:
                current_best_val_accu = val_accu
                patience_counter = 0

                para_dict = sess.run(vgg16.para_dict)
                np.save(os.path.join(FLAG.save_dir, "para_dict.npy"), para_dict)
                print("save in %s" % os.path.join(FLAG.save_dir, "para_dict.npy"))
            else:
                patience_counter += 1

            # shuffle Xtrain and Ytrain in the next epoch
            idx = np.random.permutation(Xtrain.shape[0])
            Xtrain, Ytrain = Xtrain[idx,:,:,:], Ytrain[idx,:]

            # epoch end
            # writer.add_summary(epoch_summary, epoch_counter)
            epoch_counter += 1

            ptrain.value = 0
            pval.value = 0
            bar_train.finish()
            bar_val.finish()

            print("Epoch %s (%s), %s sec >> train loss: %.4f, train accu: %.4f, val loss: %.4f, val accu: %.4f" % (epoch_counter, patience_counter, round(time.time()-stime,2), train_loss, train_accu, val_loss, val_accu))
        saver.save(sess, checkpoint_path, global_step=epoch_counter)

    FLAG.optimizer = opt_type
    FLAG.lr = start_learning_rate
    FLAG.batch_size = batch_size
    FLAG.epoch_end = epoch_counter
    FLAG.val_accu = current_best_val_accu

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
    if os.path.exists("/home/cmchang/new_CP_CNN/model.csv"):
        with open("/home/cmchang/new_CP_CNN/model.csv", "a") as myfile:
            myfile.write(row)
    else:
        with open("/home/cmchang/new_CP_CNN/model.csv", "w") as myfile:
            myfile.write(header)
            myfile.write(row)

def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars): 
            sess.run(tf.variables_initializer(not_initialized_vars))

if __name__ == '__main__':
    main()
