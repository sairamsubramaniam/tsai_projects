
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
    #layers.append( nn.BatchNorm2d(num_features=ocl[0]) )
    #layers.append( nn.ReLU() )

    for ot in range(len(ocl)-1):
        layers.append( conv(c_in=ocl[ot], c_out=ocl[ot+1], padding=pl[ot+1]) )
        #layers.append( nn.BatchNorm2d(num_features=ocl[ot+1]) )
        #layers.append( nn.ReLU() )


    return nn.Sequential(*layers)



def conv_mp_bn_relu(c_in, c_out, k=3, padding=0, padding_mode="replicate"):
    return nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, 
                              padding=padding, padding_mode=padding_mode),
                    nn.MaxPool2d(2, 2),
                    nn.BatchNorm2d(num_features=c_out),
                    nn.ReLU()
           )



def conv_resnet_block(c_in, c_out):
    return nn.Sequential(
            conv_mp_bn_relu(c_in=c_in, c_out=c_out, k=3, padding=1),
            conv_block(c_in=c_out, out_channel_list=[c_out,c_out], padding_list=[1,1])
            )





class NewResNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.prep = conv_block(c_in=3, out_channel_list=[64], padding_list=[1])

        self.input1 = conv_mp_bn_relu(c_in=64, c_out=128, k=3, padding=1)
        self.layer1 = conv_block(c_in=128, out_channel_list=[128,128], padding_list=[1,1])

        self.layer2 = conv_mp_bn_relu(c_in=128, c_out=256, k=3, padding=1)

        self.input3 = conv_mp_bn_relu(c_in=256, c_out=512, k=3, padding=1)
        self.layer3 = conv_block(c_in=512, out_channel_list=[512,512], padding_list=[1,1])

        self.maxpool = nn.MaxPool2d(4, 4)

        self.fc = nn.Linear(in_features=512, out_features=10)


    def forward(self, data_in):
        out_prep = self.prep(data_in)          # 32

        out_input1 = self.input1(out_prep)     # 16
        out_layer1 = self.layer1(out_input1)   # 16
        out_id1 = out_input1 + out_layer1      # 16

        out_layer2 = self.layer2(out_id1)      #  8

        out_input3 = self.input3(out_layer2)   #  4
        out_layer3 = self.layer3(out_input3)   #  4
        out_id3 = out_input3 + out_layer3      #  4

        out_mp = self.maxpool(out_id3)         #  1
        out_mp = out_mp.reshape(-1, 512)

        out_fc = self.fc(out_mp)

        return F.softmax(out_fc, dim=1)




