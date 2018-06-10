import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from reader import readShortVideo, getVideoList

def one_hot_encoding(arr, num_classes):
    res = np.zeros((arr.size, num_classes))
    res[np.arange(arr.size),arr] = 1
    return(res)

def load_frame(pkl_file):
    return pickle.load(open(pkl_file, "rb"))

def load_aggregate_frame(pkl_file):
    res = list()
    codes = load_frame(pkl_file)
    for i in range(len(codes)):
        res.append(np.mean(codes[i], axis=0))
    return np.array(res)

def load_extract_video(video_path, df, model, filename):
    print("===== read video =====")
    codes = list()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        for i in range(df.shape[0]):
            print(i, end="\r")
            video = readShortVideo(video_path=video_path,
                                   video_category=df.iloc[i]['Video_category'],
                                   video_name=df.iloc[i]['Video_name'],
                                   downsample_factor=12,
                                   rescale_factor=1)
            
            # extract features batch-wise
            if video.shape[0] < 50:
                tmp = sess.run(model.output, feed_dict={model.x:video})
            else:
                tmp = list()
                for i in range(int(video.shape[0]/50)+1):
                    st = 50*i
                    ed = min(50*i+50, video.shape[0])
                    tmp_video = video[st:ed,:]
                    tmp.append(sess.run(model.output, feed_dict={model.x:tmp_video}))
                tmp = np.concatenate(tmp, axis=0)
            codes.append(tmp)
    print('Done')
    
    print("===== save into %s =====" % filename)
    with open(filename, 'wb') as f:
        pickle.dump(codes, f)

def pad_feature_maxlen(feature_list, max_len):
    pad_feature_list = list()
    seq_end_index = list()
    for feature in feature_list:
        feature_len = feature.shape[0]
        if feature_len < max_len:
            padded = np.zeros([max_len-feature_len, feature.shape[1]])
            seq_end_index.append(feature_len) # due to zero-indexing in Python
            pad_feature_list.append(np.concatenate([feature, padded]))
        else:
            sample_index = np.linspace(0, feature_len-1, num=max_len, dtype=np.int32)
            seq_end_index.append(max_len) # due to zero-indexing in Python
            pad_feature_list.append(feature[sample_index,:])
    
    return np.array(pad_feature_list), np.array(seq_end_index)

