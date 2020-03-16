# this code is for checking the code 
import numpy as np
import torch
#import cv2
from torch.utils.data import Dataset, DataLoader

from  data.random_data_loader import RandomDataset 
from models.VONet import DVOFeature,DVORegression
from models.discriminator import Discriminator
from utils.options import parse as parse
from train import train,test
def test_train():
    args = parse()
    print(args)
    motion_ax_i = [int(i) for i in args.motion_ax.split(' ')]
    dataset = RandomDataset(100,motion_ax = motion_ax_i)
    dataloader = DataLoader(dataset, batch_size=1,shuffle=False ,num_workers=1,drop_last=True,worker_init_fn=lambda wid:np.random.seed(np.uint32(torch.initial_seed() + wid)))
    dvo_feature_extractor = DVOFeature()
    dvo_regressor         = DVORegression()
    dvo_discriminator     = Discriminator(500,500,2)
    trained_feature,trained_regressor = train(dvo_feature_extractor,dvo_regressor,dataloader,args)
    torch.save(trained_feature.state_dict(),'feature_seed'+args.motion_ax.replace(' ','')+str(args.epoch)+'.pt')
    torch.save(trained_regressor.state_dict(),'regressor_seed'+args.motion_ax.replace(' ','')+str(args.epoch)+'.pt')

def test_test():
    args = parse()
    print(args)
    motion_ax_i = [int(i) for i in args.motion_ax.split(' ')]
    test_motion_ax_i = [int(i) for i in args.test_motion_ax.split(' ')]
    dataset = RandomDataset(2,motion_ax = test_motion_ax_i)
    dataloader = DataLoader(dataset, batch_size=1,shuffle=False ,num_workers=1,drop_last=True)
    dvo_feature_extractor = DVOFeature()
    dvo_regressor         = DVORegression()
    dvo_discriminator     = Discriminator(500,500,2)
    dvo_feature_extractor.load_state_dict(torch.load('feature'+args.motion_ax.replace(' ','')+'.pt'))
    dvo_regressor.load_state_dict(torch.load('regressor'+args.motion_ax.replace(' ','')+'.pt'))
    test(dvo_feature_extractor,dvo_regressor,dataloader,args)


def test_data():
    dataset = RandomDataset(10,motion_ax=[0,0,1,0,1,0])
    dataloader = DataLoader(dataset, batch_size=4,shuffle=False ,num_workers=1,drop_last=True)
    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched['image_f_01'])
        print(i_batch, sample_batched['motion_f_01'])
    img_1,img_2,depth,motion = dataset.show_item()
    print(motion)
    #cv2.imshow('img_1',img_1)
    #cv2.imshow('img_2',img_2)
    #cv2.imshow('depth',depth/10)
    #cv2.waitKey()

    

def test_model(image):
    dvo_feature_extractor = DVOFeature()
    dvo_regressor         = DVORegression()
    dvo_discriminator     = Discriminator(500,500,2)
    feature = dvo_feature_extractor(image)
    print(feature.shape)
    motion  = dvo_regressor(feature)
    print(motion.shape)
    dis     = dvo_discriminator(feature)
    print(dis)

def main():
    image = np.ones((1,6,480,640))
    image = torch.Tensor(image)
    #test_model(image)
    #test_data()
    #test_test()
    test_train()


if __name__ == '__main__':
    main()

