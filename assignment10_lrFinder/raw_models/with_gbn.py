
"""
Source for Ghost Batch Norm: https://github.com/apple/ml-cifar-10-faster/blob/master/utils.py#L138
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, 
                                                                  self.num_features), 
                                           dim=0
                                 ).repeat(self.num_splits
                                 )

            self.running_var = torch.mean(self.running_var.view(self.num_splits, 
                                                                self.num_features), 
                                          dim=0
                                ).repeat(self.num_splits
                                )

        return super().train(mode)

    def forward(self, input):

        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(input.view(-1, C * self.num_splits, H, W), 
                                self.running_mean, 
                                self.running_var,
                                self.weight.repeat(self.num_splits), 
                                self.bias.repeat(self.num_splits),
                                True, 
                                self.momentum, 
                                self.eps
                   ).view(N, C, H, W
                   )
        
        else:
            return F.batch_norm(input, 
                                self.running_mean[:self.num_features], 
                                self.running_var[:self.num_features],
                                self.weight, 
                                self.bias, 
                                False, 
                                self.momentum, 
                                self.eps
                   )



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, bias=False)
        self.bn1 = GhostBatchNorm(8, 4)
        self.conv2 = nn.Conv2d(8, 8, 3, bias=False)
        self.bn2 = GhostBatchNorm(8, 4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.antman = nn.Conv2d(8, 8, 1, bias=False)
        self.conv3 = nn.Conv2d(8, 16, 3, bias=False)
        self.bn3 = GhostBatchNorm(16, 4)
        self.conv4 = nn.Conv2d(16, 16, 3, bias=False)
        self.bn4 = GhostBatchNorm(16, 4)
        self.conv5 = nn.Conv2d(16, 16, 3, bias=False)
        self.bn5 = GhostBatchNorm(16, 4)
        self.conv6 = nn.Conv2d(16, 10, 3, bias=False)
        self.bn6 = GhostBatchNorm(10, 4)
        self.gap = nn.AvgPool2d(4)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.antman(self.pool1(x))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.gap(self.bn6(self.conv6(x)))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)



