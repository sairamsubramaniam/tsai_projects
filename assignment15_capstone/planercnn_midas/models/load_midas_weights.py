
import os
import pickle

import torch


def load_midas_weights(model):

    print(os.listdir("."))

    # MODELS & WEIGHTS - MIDAS
    midas_scratch_weights_path = '../drive/My Drive/eva_stored_from_colab/eva5/s15_capstone/midas_scratch_list_of_tuples.pkl'

    with open(midas_scratch_weights_path, "rb") as pklfl:
        mds_wts = pickle.load(pklfl)

    for nm, wt in mds_wts:
        exec("model.midas_scratch." + nm + ".weight.data = torch.tensor(wt)")

    model = model.cuda()

    return model

