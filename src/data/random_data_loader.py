import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils.inverse_warp import inverse_warp 


class RandomDataset(Dataset):
    def __init__(self,data_length=1000, transform_=None,camera_parameter=[64,48,32,32,32,24]):
        self.data_length = data_length
        self.camera_parameter = camera_parameter
        self.camera_intrinsic =torch.Tensor([self.camera_parameter[2],0,self.camera_parameter[4],0,self.camera_parameter[3],self.camera_parameter[5],0,0,1]).view(3,3)
        self.inverse_camera_intrinsic = self.camera_intrinsic.inverse()

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        image_0 = torch.Tensor(np.random.random((3,self.camera_parameter[1],self.camera_parameter[0])))#image_0
        depth_0 = torch.Tensor(np.random.random((self.camera_parameter[1],self.camera_parameter[0])))#depth
        motion_se = torch.Tensor(np.random.random((6)))#motion
        image_1= self.warp(image_0,depth_0,motion_se)
        image_f_01 = torch.Tensor( np.concatenate((image_0,image_1),axis=0))
        sample = {'image_f_01':image_f_01,'motion_f_01':motion_se}
        return sample

    def warp(self,image,depth,motion_se):
        return inverse_warp(image.unsqueeze(0),depth.unsqueeze(0),\
                motion_se.unsqueeze(0),self.camera_intrinsic.unsqueeze(0))[0].squeeze()


def main():
    kitti_dataset = RandomDataset(10)
    print(len(kitti_dataset))
    dataloader = DataLoader(kitti_dataset, batch_size=4,shuffle=False ,num_workers=1,drop_last=True)
    for i_batch, sample_batched in enumerate(dataloader):
	    print(i_batch, sample_batched['image_f_01'])
	    print(i_batch, sample_batched['motion_f_01'])
if __name__== '__main__':
    main()
