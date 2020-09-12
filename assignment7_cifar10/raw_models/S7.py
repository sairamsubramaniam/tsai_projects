
import torch
import torch.nn as nn
import torch.nn.functional as F



def conv(c_in, c_out, k=3, padding=0, padding_mode="replicate"):
    return nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, 
                              padding=padding, padding_mode=padding_mode),
                    nn.BatchNorm2d(num_features=c_out),
                    nn.ReLU()
           )


def conv_depthwiseSep(c_in, c_out, k=3, padding=0, padding_mode="replicate"):
    return nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=k, groups=c_in,
                              padding=padding, padding_mode=padding_mode),
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1,
                              padding=padding, padding_mode=padding_mode),
                    nn.BatchNorm2d(num_features=c_out),
                    nn.ReLU()
           )


def transition_block(c_in, c_out):
    return nn.Sequential(
                nn.MaxPool2d(2, 2),
                conv(c_in=c_in, c_out=c_out, k=1)
           )


def conv_block(c_in, out_channel_list, padding_list=None):

    ocl = out_channel_list
    pl = padding_list
    if not padding_list:
        padding_list = [0]*len(out_channel_list)
    else:
        if len(padding_list) != len(out_channel_list):
            raise Exception("Length of padding_list and out_channel_list should be the same")

    layers = [ conv(c_in=c_in, c_out=ocl[0], padding=padding_list[0]) ]
    layers.append( nn.BatchNorm2d(num_features=ocl[0]) )
    layers.append( nn.ReLU() )

    for ot in range(len(ocl)-1):
        layers.append( conv(c_in=ocl[ot], c_out=ocl[ot+1], padding=pl[ot+1]) )
        layers.append( nn.BatchNorm2d(num_features=ocl[ot+1]) )
        layers.append( nn.ReLU() )


    return nn.Sequential(*layers)



class S7(nn.Module):

    def __init__(self):

        super().__init__()

        self.c1 = conv_block(c_in=3, out_channel_list=[16, 32], padding_list=[1, 1])
        self.t1 = transition_block(c_in=32, c_out=32)

        self.c2 = conv_block(c_in=32, out_channel_list=[64], padding_list=[1])
        self.dws = conv_depthwiseSep(c_in=64, c_out=128, padding=1)
        self.dil1 = nn.Conv2d(in_channels=64, 
                              out_channels=128, 
                              kernel_size=3,
                              dilation=2,
                              padding=2, padding_mode="replicate")
        self.t2 = transition_block(c_in=128, c_out=256)
        
        self.c3 = conv_block(c_in=256, out_channel_list=[512], padding_list=[1])
        #self.t3 = transition_block(c_in=512, c_out=256)

        self.last = nn.Conv2d(in_channels=512, out_channels=10, kernel_size=3, padding=1)
        self.gap = nn.AvgPool2d(kernel_size=8)
        


    def forward(self, data_in):
        out_c1 = self.c1(data_in)       # 36, 36
        out_t1 = self.t1(out_c1)        # 16, 16
        out_c2 = self.c2(out_t1)        # 16
        out_dws = self.dws(out_c2)      # 16, 16
        out_dil1 = self.dil1(out_c2)    # 16
        dws_dil1 = out_dil1 + out_dil1  # 16
        out_t2 = self.t2(dws_dil1)      # 8
        out_c3 = self.c3(out_t2)        # 8
        out_last = self.last(out_c3)    # 8
        out_gap = self.gap(out_last)
        data_out = out_gap.view(-1, 10)
        return F.log_softmax(data_out, dim=1)




