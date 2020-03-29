import sys
import torch
import numpy as np
import imageio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils.inverse_warp import inverse_warp 


class RandomDataset(Dataset):
    def __init__(self,data_length=1000, transform_=None,camera_parameter=[640,480,320,320,320,240],motion_ax=[1,1,1,1,1,1]):
        print(motion_ax)
        self.data_length = data_length
        self.camera_parameter = camera_parameter
        self.camera_intrinsic =torch.Tensor([self.camera_parameter[2],0,self.camera_parameter[4],0,self.camera_parameter[3],self.camera_parameter[5],0,0,1]).view(3,3)
        self.inverse_camera_intrinsic = self.camera_intrinsic.inverse()
        self.motion_ax = np.array(motion_ax)
        self.motion_ax_normalize = self.motion_ax.copy() 
        for i in range(len(self.motion_ax)):
            if self.motion_ax_normalize[i]!=0:
                self.motion_ax_normalize[i]=1/self.motion_ax_normalize[i]
        self.motion_ax_normalize = torch.Tensor(self.motion_ax_normalize)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        image_0 = torch.Tensor(np.random.random((3,self.camera_parameter[1],self.camera_parameter[0])))#image_0
        image_c = np.random.random((3,self.camera_parameter[1],self.camera_parameter[0]))#image_0
        depth_0 = torch.Tensor(np.random.random((self.camera_parameter[1],self.camera_parameter[0])))#depth
        motion_se = torch.Tensor((2*np.random.random((6))-1)*self.motion_ax)#motion
        image_1= self.warp(image_0,image_c,depth_0,motion_se)
        image_f_01 = torch.Tensor( np.concatenate((image_0,image_1),axis=0))

        sample = {'image_f_01':image_f_01,'motion_f_01':motion_se*self.motion_ax_normalize}
        return sample
    def show_item(self):
        image_0 = np.random.random((3,self.camera_parameter[1],self.camera_parameter[0]))#image_0
        image_c = np.random.random((3,self.camera_parameter[1],self.camera_parameter[0]))#image_0
        depth_0 = 100*np.random.random((self.camera_parameter[1],self.camera_parameter[0]))#depth
        motion_se = (2*np.random.random((6))-1)*self.motion_ax#motion
        image_1= self.warp_np(image_0,image_c,depth_0,motion_se)
        return image_0.transpose(1,2,0),image_1.transpose(1,2,0),depth_0,motion_se

    def warp_np(self,image,depth,motion_se):
        image_1,valid= inverse_warp(torch.Tensor(image).unsqueeze(0),torch.Tensor(depth).unsqueeze(0),\
                torch.Tensor(motion_se).unsqueeze(0),self.camera_intrinsic.unsqueeze(0)) 
        image_1 = image_1.squeeze()
        valid   = valid.squeeze()
        image_1= inverse_warp(torch.Tensor(image).unsqueeze(0),torch.Tensor(depth).unsqueeze(0),\
                torch.Tensor(motion_se).unsqueeze(0),self.camera_intrinsic.unsqueeze(0))[1].squeeze()
        return np.array(image_1)

    def warp(self,image,image_c,depth,motion_se):
        image_1,valid= inverse_warp(torch.Tensor(image).unsqueeze(0),torch.Tensor(depth).unsqueeze(0),\
                torch.Tensor(motion_se).unsqueeze(0),self.camera_intrinsic.unsqueeze(0)) 
        image_1 = image_1.squeeze()
        valid   = valid.squeeze()
        return image_1+torch.Tensor(image_c)*(1-valid.float())
 
def load_image(image_path):
    image_file = open(image_path)
    image_list = image_file.read().split('\n')[1:]
    return image_list

class RandomDatasetAdv(Dataset):
    def __init__(self,data_length=1000,
    transform_=None,camera_parameter=[640,480,320,320,320,240],motion_path=None,image_path=None,motion_ax=[1,1,1,1,1,1]):
        print(motion_ax)
        self.data_length = data_length
        self.camera_parameter = camera_parameter
        self.camera_intrinsic =torch.Tensor([self.camera_parameter[2],0,self.camera_parameter[4],0,self.camera_parameter[3],self.camera_parameter[5],0,0,1]).view(3,3)
        self.inverse_camera_intrinsic = self.camera_intrinsic.inverse()
        self.motion_ax = np.array(motion_ax)
        self.motion = None
        self.image  = None
        if motion_path is not None:
            self.motion = np.loadtxt(motion_path)
            self.data_length = self.motion.shape[0]
        if image_path is not None:
            self.image = load_image(image_path)

        self.motion_ax_normalize = self.motion_ax.copy() 
        for i in range(len(self.motion_ax)):
            if self.motion_ax_normalize[i]!=0:
                self.motion_ax_normalize[i]=1/self.motion_ax_normalize[i]
        self.motion_ax_normalize = torch.Tensor(self.motion_ax_normalize)
    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        image_0 = torch.Tensor(np.random.random((3,self.camera_parameter[1],self.camera_parameter[0])))#image_0
        depth_0 = torch.Tensor(np.random.random((self.camera_parameter[1],self.camera_parameter[0])))#depth
        motion_se = torch.Tensor((2*np.random.random((6))-1)*self.motion_ax)#motion
        if self.motion is not None:
            motion_se = torch.Tensor(self.motion[idx,:])
        if self.image is not None:
            image_0 = imageio.imread(self.image[idx])
            image_0 = torch.Tensor(image_0[:,:,:3].transpose(2,0,1)/255)

        image_1= self.warp(image_0,depth_0,motion_se)
        image_f_01 = torch.Tensor( np.concatenate((image_0,image_1),axis=0))
        sample = {'image_f_01':image_f_01,'motion_f_01':motion_se*self.motion_ax_normalize}
        return sample
    def show_item(self,idx):
        image_0 = np.random.random((3,self.camera_parameter[1],self.camera_parameter[0]))#image_0
        depth_0 = 100*np.random.random((self.camera_parameter[1],self.camera_parameter[0]))#depth
        image_1_c = np.random.random((3,self.camera_parameter[1],self.camera_parameter[0]))#image_0
        motion_se = (2*np.random.random((6))-1)*self.motion_ax#motion
        if self.motion is not None:
            motion_se = self.motion[idx,:]*self.motion_ax
        if self.image is not None:
            image_0 = imageio.imread(self.image[idx])
            image_0 = image_0[:,:,:3].transpose(2,0,1)/255
        image_1= self.warp_np(image_0,image_1_c,depth_0,motion_se)
        print(image_0.shape,image_1.shape)
        return image_0.transpose(1,2,0),image_1.transpose(1,2,0),depth_0,motion_se

    def warp_np(self,image,image_c,depth,motion_se):
        image_1,valid= inverse_warp(torch.Tensor(image).unsqueeze(0),torch.Tensor(depth).unsqueeze(0),\
                torch.Tensor(motion_se).unsqueeze(0),self.camera_intrinsic.unsqueeze(0)) 
        image_1 = image_1.squeeze()
        valid   = valid.squeeze()
        image_1 = image_1+torch.Tensor(image_c)*(1-valid.float())
        return np.array(image_1)

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
