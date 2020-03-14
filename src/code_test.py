# this code is for checking the code 
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader

from  data.random_data_loader import RandomDataset 
from models.VONet import DVOFeature,DVORegression
from models.discriminator import Discriminator
from train import train
def test_train():
    dataset = RandomDataset(100)
    dataloader = DataLoader(dataset, batch_size=1,shuffle=False ,num_workers=1,drop_last=True)
    dvo_feature_extractor = DVOFeature()
    dvo_regressor         = DVORegression()
    dvo_discriminator     = Discriminator(500,500,2)
    train(dvo_feature_extractor,dvo_regressor,dataloader)

def test_data():
    dataset = RandomDataset(10,motion_ax=[0,0,1,0,0,0])
    dataloader = DataLoader(dataset, batch_size=4,shuffle=False ,num_workers=1,drop_last=True)
    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched['image_f_01'])
        print(i_batch, sample_batched['motion_f_01'])
    img_1,img_2,depth,motion = dataset.show_item()
    print(motion)
    cv2.imshow('img_1',img_1)
    cv2.imshow('img_2',img_2)
    cv2.imshow('depth',depth/10)
    cv2.waitKey()

    

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
    test_train()


if __name__ == '__main__':
    main()

