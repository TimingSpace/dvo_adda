# This code is based on https://github.com/ClementPinard/SfmLearner-Pytorch
import torch
import torch.nn as nn


def conv(in_planes, out_planes, kernel_size=3,stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride),

        #nn.SyncBatchNorm(out_planes),
        nn.GroupNorm(16,out_planes),
        #nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )
def dia_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2,stride=2,dilation = 2),

        #nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


# Feature extraction for visual odometry
class DVOFeature(nn.Module):
    def __init__(self):
        super(DVOFeature, self).__init__()
        conv_planes =     [16, 32, 64, 128, 256]
        #base
        self.conv1 = conv(6, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3],stride=1)
        self.conv5 = conv(conv_planes[3], conv_planes[4],stride=2)
        #vo
        self.vonet=nn.Sequential(self.conv1,self.conv2,self.conv3,self.conv4,\
                self.conv5)
        self.fc1   = nn.Linear(256*30*40,500) # to detemine the width and height

    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m,nn.Linear):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_pairs):
        conv_out = self.vonet(image_pairs)
        return self.fc1(conv_out.view(conv_out.shape[0],256*30*40))# never use -1
# visual odometry regression
class DVORegression(nn.Module):
    def __init__(self):
        super(DVORegression, self).__init__()
        self.fc2 = nn.Linear(500,6)

    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m,nn.Linear):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feature):
        return 0.01*self.fc2(feature)


'''
## return the feature map for visualization
class PADVOFeature(nn.Module):

    def __init__(self,coor_layer_flag=True):
        super(PADVOFeature, self).__init__()

        conv_planes =     [16, 32, 64, 128, 256, 512]
        att_conv_planes = [16, 32, 64, 128, 256, 16]
        #base
        self.conv1 = conv(8, conv_planes[0], kernel_size=7)
        if coor_layer_flag==False:
            self.conv1 = conv(6, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3],stride=1)
        self.conv5 = conv(conv_planes[3], conv_planes[4],stride=2)

        #vo
        self.conv6 = conv(conv_planes[4], conv_planes[5],stride=1)
        self.vo_pred = nn.Conv2d(conv_planes[5], 6, kernel_size=1, padding=0)

        self.vonet=nn.Sequential(self.conv1,self.conv2,self.conv3,self.conv4,\
                self.conv5,self.conv6,self.vo_pred)

        # attention
        #self.att_conv3 = conv(att_conv_planes[1], att_conv_planes[2])
        #self.att_conv4 = conv(att_conv_planes[2], att_conv_planes[3],stride=1)
        #self.att_conv5 = conv(att_conv_planes[3], att_conv_planes[4],stride=2)
        self.att_conv6 = conv(att_conv_planes[4], att_conv_planes[5],stride=1)
        self.att_pred = nn.Conv2d(att_conv_planes[5], 1, kernel_size=1, padding=0)
        #self.soft_max = nn.Softmax(2)
        #self.hard_max = HardMax()

        self.attnet = nn.Sequential(self.conv1,self.conv2,self.conv3,self.conv4,\
                self.conv5,self.att_conv6,self.att_pred)

    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTransvo2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_pairs):
        input = image_pairs
        f1 = self.conv1(input)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f5 = self.conv5(f4)
        v6 = self.conv6(f5)
        a6 = self.att_conv6(f5)
        vo = self.vo_pred(v6)
        att=self.att_pred(a6)
        return f1,f2,f3,f4,f5,v6,a6,vo,att
## SFMVO
class SFMVONet(nn.Module):

    def __init__(self,coor_layer_flag=True):
        super(SFMVONet, self).__init__()

        conv_planes =     [16, 32, 64, 128, 256, 512]
        att_conv_planes = [16, 32, 64, 128, 256, 16]
        #base
        self.conv1 = conv(8, conv_planes[0], kernel_size=7)
        if coor_layer_flag==False:
            self.conv1 = conv(6, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3],stride=1)
        self.conv5 = conv(conv_planes[3], conv_planes[4],stride=2)

        #vo
        self.conv6 = conv(conv_planes[4], conv_planes[5],stride=1)
        self.vo_pred = nn.Conv2d(conv_planes[5], 6, kernel_size=1, padding=0)

        self.vonet=nn.Sequential(self.conv1,self.conv2,self.conv3,self.conv4,\
                self.conv5,self.conv6,self.vo_pred)

        # attention
        #self.att_conv3 = conv(att_conv_planes[1], att_conv_planes[2])
        #self.att_conv4 = conv(att_conv_planes[2], att_conv_planes[3],stride=1)
        #self.att_conv5 = conv(att_conv_planes[3], att_conv_planes[4],stride=2)
        self.att_conv6 = conv(att_conv_planes[4], att_conv_planes[5],stride=1)
        self.att_pred = nn.Conv2d(att_conv_planes[5], 1, kernel_size=1, padding=0)
        #self.soft_max = nn.Softmax(2)
        #self.hard_max = HardMax()

        self.attnet = nn.Sequential(self.conv1,self.conv2,self.conv3,self.conv4,\
                self.conv5,self.att_conv6,self.att_pred)

    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTransvo2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_pairs):
        input = image_pairs
        vo  = self.vonet(input)
        att = self.attnet(input)
        return vo,att



## Current
class PADVONet(nn.Module):

    def __init__(self,coor_layer_flag=True):
        super(PADVONet, self).__init__()

        conv_planes =     [16, 32, 64, 128, 256, 512]
        att_conv_planes = [16, 32, 64, 128, 256, 16]
        #base
        self.conv1 = conv(8, conv_planes[0], kernel_size=7)
        if coor_layer_flag==False:
            self.conv1 = conv(6, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3],stride=1)
        self.conv5 = conv(conv_planes[3], conv_planes[4],stride=2)

        #vo
        self.conv6 = conv(conv_planes[4], conv_planes[5],stride=1)
        self.vo_pred = nn.Conv2d(conv_planes[5], 6, kernel_size=1, padding=0)

        self.vonet=nn.Sequential(self.conv1,self.conv2,self.conv3,self.conv4,\
                self.conv5,self.conv6,self.vo_pred)

        # attention
        #self.att_conv3 = conv(att_conv_planes[1], att_conv_planes[2])
        #self.att_conv4 = conv(att_conv_planes[2], att_conv_planes[3],stride=1)
        #self.att_conv5 = conv(att_conv_planes[3], att_conv_planes[4],stride=2)
        self.att_conv6 = conv(att_conv_planes[4], att_conv_planes[5],stride=1)
        self.att_pred = nn.Conv2d(att_conv_planes[5], 1, kernel_size=1, padding=0)
        #self.soft_max = nn.Softmax(2)
        #self.hard_max = HardMax()

        self.attnet = nn.Sequential(self.conv1,self.conv2,self.conv3,self.conv4,\
                self.conv5,self.att_conv6,self.att_pred)

    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTransvo2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_pairs):
        input = image_pairs
        vo  = self.vonet(input)
        att = self.attnet(input)
        return vo,att

'''
