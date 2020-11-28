"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import pickle

import torch
from torch import optim
from torch.utils.data import DataLoader

import os
from tqdm import tqdm
import numpy as np
import cv2
import sys

from models.all_in_one import MidasScratch
from models.model import *
from models.refinement_net import *
from models.modules import *
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from datasets.plane_stereo_dataset import *

from utils import *
from visualize_utils import *
from evaluate_utils import *
from options import parse_args
from config import PlaneConfig



def load_models(options):

    config = PlaneConfig(options)
    
    # MODEL & WEIGHTS - PLANERCNN
    model = MaskRCNN(config)
    refine_model = RefineModel(options)
    model.cuda()
    model.train()    
    refine_model.cuda()
    refine_model.train()

    if options.restore == 1:
        ## Resume training
        print('restore')
        model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'))
        refine_model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_refine.pth'))
    elif options.restore == 2:
        ## Train upon Mask R-CNN weights
        model_path = options.MaskRCNNPath
        print("Loading pretrained weights ", model_path)
        model.load_weights(model_path)
        pass


    # MODELS & WEIGHTS - MIDAS
    midas_scratch_weights_path = '../drive/My Drive/eva_stored_from_colab/eva5/s15_capstone/midas_scratch_list_of_tuples.pkl'

    with open(midas_scratch_weights_path, "rb") as pklfl:
        mds_wts = pickle.load(pklfl)
    midas_scratch = MidasScratch()
    # midas_scratch.load_state_dict(torch.load(midas_scratch_weights_path))

    for nm, wt in mds_wts:
        exec("midas_scratch." + nm + ".weight.data = torch.as_tensor(wt)")



    return model, refine_model, midas_scratch
