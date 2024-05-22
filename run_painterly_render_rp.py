# -*- coding: utf-8 -*-
# Author: ximing
# Description: the main func of this project.
# Copyright (c) 2023, XiMing Xing.
# License: MIT License

import os
import sys
import argparse
from datetime import datetime
import random
from typing import Any, List
from functools import partial

from accelerate.utils import set_seed
import omegaconf
from torchvision.utils import save_image
from datetime import datetime
import glob, os, sys, pdb, time
import pandas as pd
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from imageio import imread
from PIL import Image, ImageOps
import torchvision.models as models 
import torch.nn as nn
from matplotlib import pyplot as plt
import csv
import pandas as pd
import PIL.Image as pilimg

import numpy as np
from numpy.core.fromnumeric import mean
import torch.utils.data as data
import torchvision.transforms as transforms
import sysconfig

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from libs.engine import merge_and_update_config
from libs.utils.argparse import accelerate_parser, base_data_parser

import sys
sys.path.append("/content/CLIPasso/diffvg/build/lib.linux-x86_64-3.7")
import PIL.Image as pilimg
from PIL import ImageEnhance
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.utils.data
import matplotlib

matplotlib.use('Agg')

global prompts
prompts = []


def render_batch_wrap(args: omegaconf.DictConfig,
                      seed_range: List,
                      pipeline: Any,
                      **pipe_args):
    start_time = datetime.now()
    for idx, seed in enumerate(seed_range):
        args.seed = seed  # update seed
        print(f"\n-> [{idx}/{len(seed_range)}], "
              f"current seed: {seed}, "
              f"current time: {datetime.now() - start_time}\n")
        pipe = pipeline(args)
        pipe.painterly_rendering(10, **pipe_args)


def main(args, seed_range):
    args.batch_size = 1  # rendering one SVG at a time
    all_classes = ['No Finding','Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Fracture', 'Support Devices']
    
    # partial 함수는 특정 함수의 매게변수를 고정하기 위해서 사용하는 듯
    render_batch_fn = partial(render_batch_wrap, args=args, seed_range=seed_range)
    result_dir = 'output_sketches/RP/'
    csv_path = "/home/hbc/DS/DiffSketcher/train.csv" ####
    os.makedirs(result_dir, exist_ok=True)
    all_data = pd.read_csv(csv_path)
    indices = list(range(0,1000))
    selecte_data = all_data.iloc[indices, :]


    if args.task == "diffsketcher":  # text2sketch
        global prompts
        from pipelines.painter.diffsketcher_pipeline_rp import DiffSketcherPipeline

        pipe = DiffSketcherPipeline(args)
        for i in range(len(selecte_data)):
            global prompts
            row = selecte_data.iloc[i, :]
            if row['Frontal/Lateral'] == 'Lateral':
                continue
            prompt = "A {} years old {} patient's chest X-ray with ".format(row['Age'], row['Sex'])
            for d in range(len(all_classes)):
                if row[d] == 1:
                    
                    prompt += (all_classes[d] + ', ')
            prompt = prompt[:-2]
            prompts.append(prompt)
            print(prompt)
            with open(result_dir +'prompts.txt', 'a') as file:
                file.write(prompt + '\n')
            img = pilimg.open(row['Path'])
            img = img.resize((224,224))

            img.save(result_dir + '/real_image{}.png'.format(indices[i]))
            final_raster_sketch = pipe.painterly_rendering(args.num_outputs, prompt, img, row['Path'])
            
            save_image(final_raster_sketch[0], result_dir + '/sketch{}.png'.format(indices[i]))
            # save_image(img, result_dir + '/real_image{}.png'.format(indices[i]))

        # with open(result_dir +'prompts.txt', 'a') as file:
        #         for line in prompts:
        #             file.write(line + '\n')

    elif args.task == "style-diffsketcher":  # text2sketch + style transfer
        from pipelines.painter.diffsketcher_stylized_pipeline import StylizedDiffSketcherPipeline

        if not args.render_batch:
            pipe = StylizedDiffSketcherPipeline(args)
            pipe.painterly_rendering(args.prompt, args.style_file)
        else:  # generate many SVG at once
            render_batch_fn(pipeline=StylizedDiffSketcherPipeline, prompt=args.prompt, style_fpath=args.style_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="vary style and content painterly rendering",
        parents=[accelerate_parser(), base_data_parser()] # parser를 리턴하는 함수를 호출해서 추가함. 
    )
    # flag
    parser.add_argument("-tk", "--task",
                        default="diffsketcher", type=str,
                        choices=['diffsketcher', 'style-diffsketcher'],
                        help="choose a method.")
    parser.add_argument("-n", "--num_outputs", default=1)
    parser.add_argument("-m", "--model_id", default="medsd", type=str)
    # parser.add_argument("-clip", "--clip", default="medclip", type=str)
    # config
    parser.add_argument("-c", "--config",
                        required=False, type=str,
                        default="/diffsketcher.yaml",
                        help="YAML/YML file for configuration.")
    parser.add_argument("-style", "--style_file",
                        default="", type=str,
                        help="the path of style img place.")
    # prompt
    parser.add_argument("-pt", "--prompt", default="A horse is drinking water by the lake", type=str)
    parser.add_argument("-npt", "--negative_prompt", default="", type=str)
    # DiffSVG
    parser.add_argument("--print_timing", "-timing", action="store_true",
                        help="set print svg rendering timing.")
    # diffuser
    parser.add_argument("--download", action="store_true",
                        help="download models from huggingface automatically.")
    parser.add_argument("--force_download", "-download", action="store_true",
                        help="force the models to be downloaded from huggingface.")
    parser.add_argument("--resume_download", "-dpm_resume", action="store_true",
                        help="download the models again from the breakpoint.")
    # rendering quantityl
    # like: python main.py -rdbz -srange 100 200
    parser.add_argument("--render_batch", "-rdbz", default=False , action="store_true")
    parser.add_argument("-srange", "--seed_range",
                        required=False, nargs='+',
                        help="Sampling quantity.")
    # visual rendering process
    parser.add_argument("-mv", "--make_video", action="store_true",
                        help="make a video of the rendering process.")
    parser.add_argument("-frame_freq", "--video_frame_freq",
                        default=1, type=int,
                        help="video frame control.")
    parser.add_argument("-framerate", "--video_frame_rate",
                        default=36, type=int,
                        help="by adjusting the frame rate, you can control the playback speed of the output video.")

    args = parser.parse_args()

    # set the random seed range
    seed_range = None
    if args.render_batch:
        # random sampling without specifying a range
        start_, end_ = 1, 1000000
        if args.seed_range is not None:  # specify range sequential sampling
            seed_range_ = list(args.seed_range)
            assert len(seed_range_) == 2 and int(seed_range_[1]) > int(seed_range_[0])
            start_, end_ = int(seed_range_[0]), int(seed_range_[1])
            seed_range = [i for i in range(start_, end_)]
        else:
            # a list of lengths 1000 sampled from the range start_ to end_ (e.g.: [1, 1000000]) # 이거 다양한 스캐치를 만들기 위해서 이렇게 하는 듯
            numbers = list(range(start_, end_))
            seed_range = random.sample(numbers, k=1000)

    args = merge_and_update_config(args)

    set_seed(args.seed)

    # main function 같은 경우에는 하나의 프롬프트에 대해서 스캐치를 만드는 듯. 
    main(args, seed_range)
