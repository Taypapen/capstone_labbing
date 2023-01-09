import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from pytorch_nn import Lipread2
from video_preprocess import VideoPreprocessor

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading')

    parser.add_argument('--model-path', type=str, default=None, help='Pretrained Model pathname')
    parser.add_argument('--video-path', type=str, default=None, help='Video path for prediction')
    parser.add_argument('--logging-dir', type=str, default='./train_logs',
                        help='path to the directory in which to save the log file')
    parser.add_argument('--wordlist-file', type=str, default=None, help='Path to wordlist text file')
    args = parser.parse_args()
    return args


args = load_args()


def get_wordslist_from_txt_file(file_path):
    with open(file_path) as file:
        word_list = file.readlines()
        word_list = [item.rstrip() for item in word_list]
    return word_list


def main():

    assert os.path.isfile(args.video_path), "Video file does not exist. Path input: {}".format(args.video_path)
    assert os.path.isfile(args.wordlist_file), "Word List file does not exist. Path input: {}".format(args.wordlist_file)

    #Initialize Model and load pre-trained weights
    model = Lipread2(500)
    checkpoint = torch.load(args.model_path)
    loaded_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(loaded_state_dict, strict=False)
    #Load Word list for referencing answer from prediction
    words_list = get_wordslist_from_txt_file(args.wordlist_file)

    #Load Video and set up as tensor for input
    video_setup = VideoPreprocessor()
    video_tensor, video_length = video_setup.full_tensor_setup(args.video_path)

    #Input video data into model and print word prediction
    prediction = model(video_tensor, lengths=[video_length])
    _, guess = torch.max(prediction, 1)
    print("THE WORD IS: " + words_list[guess])


main()




