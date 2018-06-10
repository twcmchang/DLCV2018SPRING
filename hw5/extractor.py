import os
import argparse
import pandas as pd
from model import Extractor
from utils import load_extract_video
from reader import getVideoList

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='HW5_data/TrimmedVideos/video/train/', help='video path')
    parser.add_argument('--video_list', type=str, default='HW5_data/TrimmedVideos/label/gt_train.csv', help='video list')   
    parser.add_argument('--init', type=str, default='updated_keras_vgg16.npy', help='initialization of extractor')
    parser.add_argument('--save_dir', type=str, default='', help='directory to save extracted features')
    parser.add_argument('--filename', type=str, default='train_code_full.pkl', help='filename of extracted features')

    FLAG = parser.parse_args()

    print("===== create directory =====")
    if not os.path.exists(FLAG.save_dir):
        os.makedirs(FLAG.save_dir)

    extract(FLAG)

def extract(FLAG):
    video_path = FLAG.video_path
    video_list = getVideoList(FLAG.video_list)

    df = pd.DataFrame.from_dict(video_list)

    vgg16 = Extractor(shape=(240,320,3))
    vgg16.build(vgg16_npy_path=FLAG.init)

    load_extract_video(video_path=video_path,df=df, model=vgg16, filename=os.path.join(FLAG.save_dir, FLAG.filename))

if __name__ == '__main__':
    main()