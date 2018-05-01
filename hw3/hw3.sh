#!/bin/bash
wget -O save_32s.zip https://www.dropbox.com/s/q97xiqzcvbazeq5/save_32s.zip?dl=0
unzip save_32s.zip -d save/
python3 test.py --test_dir $1 --plot_dir $2 --mode FCN32s --init_from save/keras-vgg16.npy --save_dir save/
