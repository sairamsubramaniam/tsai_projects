
"""
Functions helpful to analyze the behaviour of a model.
A few written by me, others lifted directly from github pages.
Sources have been mentioned wherever it was a direct copy paste.
I plan to write my own versions when I get time!
"""




# CODE below lifted directly from here `https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py`

import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]

        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if str(module) == str(self.feature_module):
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            elif "linear" in name.lower():
                x = F.avg_pool2d(x, 4)
                x = x.view(x.size(0), -1)
                x = module(x)
            else:
                x = module(x)

        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (0, 1, 2)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    # preprocessed_img = preprocessed_img.type(torch.FloatTensor)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)*2 / 255
    cam = heatmap + np.float32( torch.from_numpy(img).permute(1,2,0).numpy() )
    cam = cam / np.max(cam)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("cam.jpg", np.uint8(255 * cam))
    return cam


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    # def forward(self, input):
    #     return self.model(input)

    def __call__(self, input, index=None):

        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.feature_module.zero_grad()
        # self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


def get_grad_cam_image(img, model, feature_module, target_layers, device=None):
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    grad_cam = GradCam(model=model, feature_module=feature_module, \
                       target_layer_names=target_layers, use_cuda=True)

    #img = cv2.imread(img, 1)
    img = np.asarray(img)
    #img = np.float32(cv2.resize(img, (32, 32))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)

    return show_cam_on_image(img, mask)

    




##############################################################################################



import matplotlib.pyplot as plt
import cv2




def create_plot_pos(nrows, ncols):
    num_images = nrows * ncols
    positions = []
    for r in range(num_images):
        row = r // ncols
        col = r % ncols
        positions.append((row, col))
    return positions



def plot_misclassified(imgs, targets, preds, nrows, ncols, skip=0, 
                       plt_scaler=(2,2.5), plt_fsize=12, classes=None,
                       gradcam_params=None):
    """
    imgs is a tensor of all images
    targets is a tensor of all label targets
    preds is a tensor of all predictions from the model
    """
    matches = preds.eq(targets)

    total_imgs = nrows*ncols
    pos = create_plot_pos(nrows, ncols)

    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols, 
                             figsize=(ncols*plt_scaler[1], nrows*plt_scaler[0]), 
                             sharex=True, 
                             sharey=True)

    idx = 0
    posidx = 0
    total_skipped = 0
    for m in matches:
        if posidx > total_imgs-1:
            break

        if not m:
            if total_skipped <= skip:
                total_skipped += 1
                idx += 1
                continue

            if gradcam_params:
                img = get_grad_cam_image(img=imgs[idx], **gradcam_params)

            else:
                img = imgs[idx].permute(1,2,0)

            if classes:
                tgt = classes[targets[idx].item()]
                prd = classes[preds[idx].item()]
                title = "Act: " + str(tgt) + ", Pred: " + str(prd)
            else:
                title = "Act: " + str(targets[idx].item()) + ", Pred: " + str(preds[idx].item())

            chart_pos = pos[posidx]
            axes[chart_pos].imshow(img)
            axes[chart_pos].set_title(title, fontsize=plt_fsize)
            axes[chart_pos].axis("off")

            posidx += 1
        
        idx += 1
    
    return fig, axes


