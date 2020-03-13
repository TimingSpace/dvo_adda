import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class RandomDataset(Dataset):
	"""
	"""
	def __init__(self,data_length, transform_=None,camera_parameter=[640,480,320,320,320,240]):
		self.data_length = data_length
		self.camera_parameter = camera_parameter

	def __len__(self):
		return self.data_length

	def __getitem__(self, idx):
		image_0 = np.random.random((1,self.camera_parameter[1],self.camera_parameter[0]))#image_0
		depth_0 = np.random.random((1,self.camera_parameter[1],self.camera_parameter[0]))#depth
		motion_se = np.random.random((6))#motion
		image_1 = self.wrap(image_0,depth_0,motion_se)
		image_f_01 = torch.Tensor( np.concatenate((image_0,image_1),axis=0))
		sample = {'image_f_01':image_f_01,'motion_f_01':motion_se}
		return sample

	def wrap(self,image,depth,motion_se):
		return image


def main():
	kitti_dataset = RandomDataset(10)
	print(len(kitti_dataset))
	dataloader = DataLoader(kitti_dataset, batch_size=4,shuffle=False ,num_workers=1,drop_last=True)
	for i_batch, sample_batched in enumerate(dataloader):
		print(i_batch, sample_batched['image_f_01'])
		print(i_batch, sample_batched['motion_f_01'])
if __name__== '__main__':
	main()
