# this code is for checking the code 
import numpy as np
import torch
#import cv2
from torch.utils.data import Dataset, DataLoader

from data.random_data_loader import RandomDataset 
from data.data_loader import SepeDataset
from models.VONet import DVOFeature,DVORegression
from models.discriminator import Discriminator

from utils.options import parse as parse
from train import train,test
from adapt import adapt
import sys
from skimage import io, transform

def train_real():
    args = parse()
    print(args)
    dataset = SepeDataset(args.poses_train,args.images_train,coor_layer_flag =False)
    dataloader = DataLoader(dataset, batch_size=3,shuffle=True ,num_workers=1,drop_last=True,worker_init_fn=lambda wid:np.random.seed(np.uint32(torch.initial_seed() + wid)))
    dvo_feature_extractor = DVOFeature()
    dvo_regressor         = DVORegression()
    trained_feature,trained_regressor = train(dvo_feature_extractor,dvo_regressor,dataloader,args)
    torch.save(trained_feature.state_dict(),'feature_'+args.tag+str(args.epoch)+'.pt')
    torch.save(trained_regressor.state_dict(),'regressor_'+args.tag+str(args.epoch)+'.pt')


def adapt():
    args = parse()
    print(args)
    dataset = SepeDataset(args.poses_train,args.images_train,coor_layer_flag =False)
    dataloader = DataLoader(dataset, batch_size=1,shuffle=True ,num_workers=1,drop_last=True,worker_init_fn=lambda wid:np.random.seed(np.uint32(torch.initial_seed() + wid)))
    dataset_tgt = SepeDataset(args.poses_target,args.images_target,coor_layer_flag =False)
    dataloader_tgt = DataLoader(dataset_tgt, batch_size=1,shuffle=True ,num_workers=1,drop_last=True,worker_init_fn=lambda wid:np.random.seed(np.uint32(torch.initial_seed() + wid)))
    src_extractor = DVOFeature()
    tgt_extractor = DVOFeature()
    src_extractor.load_state_dict(torch.load(args.feature_model))
    tgt_extractor.load_state_dict(torch.load(args.feature_model))
    dvo_discriminator     = Discriminator(500,500,2)
    adapt(src_extractor,tgt_extractor,dvo_discriminator,dataloader,dataloader_tgt,args)
    torch.save(tgt_extractor.state_dict(),'tgt_feature_'+args.tag+str(args.epoch)+'.pt')
    torch.save(dvo_discriminator.state_dict(),'dis_'+args.tag+str(args.epoch)+'.pt')

def train_random():
    args = parse()
    print(args)
    motion_ax_i = [float(i) for i in args.motion_ax.split(' ')]
    dataset = RandomDataset(200,motion_ax = motion_ax_i)
    dataloader = DataLoader(dataset, batch_size=10,shuffle=False ,num_workers=1,drop_last=True,worker_init_fn=lambda wid:np.random.seed(np.uint32(torch.initial_seed() + wid)))
    dvo_feature_extractor = DVOFeature()
    dvo_regressor         = DVORegression()
    dvo_discriminator     = Discriminator(500,500,2)
    trained_feature,trained_regressor = train(dvo_feature_extractor,dvo_regressor,dataloader,[2,4],args)
    torch.save(trained_feature.state_dict(),'feature_seed'+args.tag+args.motion_ax.replace(' ','')+str(args.epoch)+'.pt')
    torch.save(trained_regressor.state_dict(),'regressor_seed'+args.tag+args.motion_ax.replace(' ','')+str(args.epoch)+'.pt')

def test_real():
    args = parse()
    print(args)
    dataset = SepeDataset(args.poses_train,args.images_train,coor_layer_flag =False)
    dataloader = DataLoader(dataset, batch_size=1,shuffle=False ,num_workers=1,drop_last=True)
    dvo_feature_extractor = DVOFeature()
    dvo_regressor         = DVORegression()
    dvo_feature_extractor.load_state_dict(torch.load(('feature'+args.tag+'.pt')))
    dvo_regressor.load_state_dict(torch.load('regressor'+args.tag+'.pt'))
    test(dvo_feature_extractor,dvo_regressor,dataloader,args)


def test_random():
    args = parse()
    print(args)
    motion_ax_i = [float(i) for i in args.motion_ax.split(' ')]
    test_motion_ax_i = [float(i) for i in args.test_motion_ax.split(' ')]
    dataset = RandomDataset(100,motion_ax = test_motion_ax_i)
    dataloader = DataLoader(dataset, batch_size=1,shuffle=False ,num_workers=1,drop_last=True)
    dvo_feature_extractor = DVOFeature()
    dvo_regressor         = DVORegression()
    dvo_discriminator     = Discriminator(500,500,2)
    dvo_feature_extractor.load_state_dict(torch.load('feature_seed'+args.tag+args.motion_ax.replace(' ','')+str(args.epoch)+'.pt'))
    dvo_regressor.load_state_dict(torch.load('regressor_seed'+args.tag+args.motion_ax.replace(' ','')+str(args.epoch)+'.pt'))
    test(dvo_feature_extractor,dvo_regressor,dataloader,[0,1,2],args)
    
def main():
    train_random()
    #test_random()
    #test_real()

if __name__ == '__main__':
    main()

