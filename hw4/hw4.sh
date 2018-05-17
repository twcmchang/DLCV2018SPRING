#!/bin/bash
wget -O vae.zip "https://www.dropbox.com/s/efcl6le9io9tpgj/save_vae.zip?dl=0"
unzip vae.zip -d save_vae/
wget -O gan.zip "https://www.dropbox.com/s/0tzn51x72s3tue0/save_gan.zip?dl=0"
unzip gan.zip -d save_gan/
wget -O acgan.zip "https://www.dropbox.com/s/ku965cvmc0n6xs0/save_acgan.zip?dl=0"
unzip acgan.zip -d save_acgan/
python3 test_vae.py --save_dir save_vae/ --data_dir $1 --plot_dir $2
python3 test_gan.py --save_dir save_gan/ --plot_dir $2
python3 test_acgan.py --save_dir save_acgan/ --plot_dir $2
