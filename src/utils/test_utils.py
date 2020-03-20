import sys
import cv2
import torch
import numpy as np
from remap import remap
from inverse_warp import inverse_warp

def test_remap():
    image = cv2.imread(sys.argv[1])
    cv2.imshow('source',image)
    source_image = torch.Tensor(image).transpose(0,2).transpose(1,2)
    intrinsics_s = torch.Tensor([[718.8560,0,607.1928],[0,718.8960,185.2152],[0,0,1]]) 
    intrinsics_t = torch.Tensor([[800,0,320],[0,800,240],[0,0,1]]) 
    target_image_shape = (640,480)
    target_image =remap(source_image.unsqueeze(0),intrinsics_s.unsqueeze(0),intrinsics_t.unsqueeze(0),target_image_shape)
    target_image_show = target_image[0].squeeze().transpose(0,2).transpose(0,1)
    cv2.imshow('target',np.array(target_image_show)/255)
    cv2.waitKey()

# 1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157
def show_item():
    image_0 = np.random.random((3,camera_parameter[1],camera_parameter[0]))#image_0
    depth_0 = 20*np.random.random((camera_parameter[1],camera_parameter[0]))#depth
    motion_se = (2*np.random.random((6))-1)*motion_ax#motion
    image_1= warp_np(image_0,depth_0,motion_se)
    return image_0.transpose(1,2,0),depth_0,image_1.transpose(1,2,0),depth_0,motion_se

def warp_np(image,depth,motion_se):
    image_1= inverse_warp(torch.Tensor(image).unsqueeze(0),torch.Tensor(depth).unsqueeze(0),\
            torch.Tensor(motion_se).unsqueeze(0),camera_intrinsic.unsqueeze(0))[0].squeeze()
    return np.array(image_1)


def test_inverse_warp():
   
def main():
    test_remap()

if __name__ == '__main__':
    main()
