#!/bin/bash
wget -O save_Q1.zip https://www.dropbox.com/s/q97xiqzcvbazeq5/save_32s.zip?dl=0
unzip save_Q1.zip -d save_Q1/
python3 extractor.py --video_path $1 --video_list $2 --filename valid_codes_full.pkl
python3 test_Q1.py --valid_video_list $2 --output_dir $3
