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

    for epoch in range(100):
        for step, samples in enumerate(dataloader):
            optimizer.zero_grad()
            images = samples['image_f_01']
            motions = samples['motion_f_01']
            feature = feature_extractor(images)
            print(feature.shape)
            motions_pred  = regressor(feature)
            loss = loss_func(motions_pred, motions)
            print('loss',loss)
            # optimize source classifier
            loss.backward()
            optimizer.step()

