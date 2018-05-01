#!/bin/bash
wget -O save_8s.zip https://www.dropbox.com/s/eaiqzffmczgq13m/save_8s.zip?dl=0
unzip save_8s.zip -d save_best/
python3 test.py --test_dir $1 --plot_dir $2 --mode FCN8s --init_from save_best/keras-vgg16.npy --save_dir save_best/
