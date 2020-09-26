
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



class Q9(nn.Module):

    def __init__(self):

        super().__init__()

        in1, out1   = 3, 16                #  3,    16
        in2, out2   = in1 + out1, 32       #  19,   32
        in3, out3   = in2 + out2, 32       #  51,   32

        in4, out4   = out3, 64             #  32,   64
        in5, out5   = in4 + out4, 128      #  96,  128
        in6, out6   = in5 + out5, 512      # 224,  512
        in7, out7   = out4+out5+out6, 256  # 704,  256

        in8, out8   = out7, 512            # 256,  512
        in9, out9   = in8 + out8, 1024     # 768, 1024
        in10, out10 = in8+out8+out9, 512   # 1792, 512


        self.c1 = conv_block(c_in=in1, out_channel_list=[16], padding_list=[1])
        self.c2 = conv_block(c_in=19, out_channel_list=[32], padding_list=[1])
        self.t1 = transition_block(c_in=51, c_out=32)

        self.c3 = conv_block(c_in=32, out_channel_list=[64], padding_list=[1])
        self.c4 = conv_block(c_in=96, out_channel_list=[128], padding_list=[1])
        self.c5 = conv_block(c_in=224, out_channel_list=[512], padding_list=[1])
        self.t2 = transition_block(c_in=704, c_out=256)

        self.c6 = conv_block(c_in=256, out_channel_list=[512], padding_list=[1])
        self.c7 = conv_block(c_in=768, out_channel_list=[1024], padding_list=[1])
        self.c8 = conv_block(c_in=1792, out_channel_list=[512], padding_list=[1])

        #self.last = nn.Conv2d(in_channels=128, out_channels=10, kernel_size=3, padding=1)
        self.gap = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(512, 10)
        


    def forward(self, data_in):
        x1 = data_in                           # 32, 3
        x2 = self.c1(x1)                       # 32, 16
        x3 = torch.cat((x1, x2), dim=1)       # 32, 19
        x3 = self.c2(x3)                       # 32, 32
        x4 = torch.cat((x1, x2, x3), dim=1)   # 32, 51
        x4 = self.t1(x4)                       # 16, 32
        x5 = self.c3(x4)                       # 16, 64
        x6 = torch.cat((x4, x5), dim=1)       # 16, 96
        x6 = self.c4(x6)                       # 16, 128
        x7 = torch.cat((x4, x5, x6), dim=1)   # 16, 224
        x7 = self.c5(x7)                       # 16, 512
        x8 = torch.cat((x5, x6, x7), dim=1)   # 16, 704
        x8 = self.t2(x8)                       #  8, 256
        x9 = self.c6(x8)                       #  8, 512
        x10 = torch.cat((x8, x9), dim=1)      #  8, 768
        x10 = self.c7(x10)                      #  8, 1024
        x11 = torch.cat((x8, x9, x10), dim=1) #  8, 1792
        x11 = self.c8(x11)                     #  8, 512
        x12 = self.gap(x11)                    #  1, 512
        x12 = torch.flatten(x12).view(-1, 512)
        x13 = self.fc(x12)

        data_out = x13.view(-1, 10)
        return F.log_softmax(data_out, dim=1)




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
        
        self.c3 = conv_block(c_in=256, out_channel_list=[256], padding_list=[1])
        self.t3 = transition_block(c_in=256, c_out=128)

        self.last = nn.Conv2d(in_channels=128, out_channels=10, kernel_size=3, padding=1)
        self.gap = nn.AvgPool2d(kernel_size=4)
        


    def forward(self, data_in):
        out_c1 = self.c1(data_in)       # 36, 36
        out_t1 = self.t1(out_c1)        # 16, 16
        out_c2 = self.c2(out_t1)        # 16
        out_dws = self.dws(out_c2)      # 16, 16
        out_dil1 = self.dil1(out_c2)    # 16
        dws_dil1 = out_dil1 + out_dil1  # 16
        out_t2 = self.t2(dws_dil1)      # 8
        out_c3 = self.c3(out_t2)        # 8
        out_t3 = self.t3(out_c3)        # 4
        out_last = self.last(out_t3)    # 4
        out_gap = self.gap(out_last)
        data_out = out_gap.view(-1, 10)
        return F.log_softmax(data_out, dim=1)




