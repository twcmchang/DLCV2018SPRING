#!/bin/bash
wget -O updated_keras_vgg16.npy https://www.dropbox.com/s/e4kifd7wpayxyzj/updated_keras_vgg16.npy?dl=0
wget -O save_Q1.zip https://www.dropbox.com/s/w32h5qkqvs3xs7a/save_Q1.zip?dl=0
unzip save_Q1.zip -d save_Q1/
python3 extractor.py --video_path $1 --video_list $2 --filename valid_codes_full.pkl --save_dir save
python3 test_Q1.py --valid_video_list $2 --output_dir $3 --valid_pkl_file save/valid_codes_full.pkl --save_dir save_Q1/
