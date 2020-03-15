import numpy as np
import torch
import cv2
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



def train(feature_extractor,regressor,dataloader,args=None):
    optimizer = optim.Adam(
        list(feature_extractor.parameters()) + list(regressor.parameters()),
                lr=0.001)
    loss_func = nn.MSELoss()
    train_log = open('train_log_'+args.motion_ax.replace(' ','')+'.txt','w')
    for epoch in range(10000):
        loss_sum = 0
        for step, samples in enumerate(dataloader):
            optimizer.zero_grad()
            images = samples['image_f_01']
            motions = samples['motion_f_01']
            feature = feature_extractor(images)
            motions_pred  = regressor(feature)
            loss = loss_func(motions_pred, motions)
            loss_sum += loss.item() 
            # optimize source classifier
            loss.backward()
            optimizer.step()
        loss_sum = loss_sum/(len(dataloader)//dataloader.batch_size)
        print('epoch  ',epoch,' loss ' ,loss_sum)
        train_log.write(str(epoch)+' '+str(loss_sum)+'\n')
        if epoch%10==1:
            train_log.flush()



