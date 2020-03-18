# this code is modified from `https://github.com/ClementPinard/SfmLearner-Pytorch`
from __future__ import division
import torch
import torch.nn.functional as F

pixel_coords = None
src_pixel_coords = None



# generate the pixel coords pixel_coords[i,j] = [i,j] with the same shape of input
def set_id_grid(size):
    global pixel_coords
    w, h = size
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).float()  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).float()  # [1, H, W]
    ones = torch.ones(1,h,w).float()
    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))

# depth image to 3d coordinates

def get_pixed_coords(intrinsics_trans,target_image_shape,source_image_shape):
    """
    get grid for remap
    Args:
    intrinsics_trans: camera intrinsic matrix from target image to sourse
    intri_s*intri_t_inv -- [B, 3, 3]
    target_image_shape: camera_shape for target image      -- (2)
    source_image_shape: camera_shape for source image      -- (2)
    """
    global pixel_coords
    w,h = target_image_shape
    w_s,h_s = source_image_shape
    b = intrinsics_trans.shape[0]
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(target_image_shape)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)  # [B, 3, H*W]
    s_coords = intrinsics_trans @ current_pixel_coords
    X = s_coords[:, 0]
    Y = s_coords[:, 1]
    Z = s_coords[:, 2].clamp(min=1e-3)
    print(Z)

    X_norm = 2*(X / Z)/(w_s-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h_s-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b,h,w,2)




def remap(img, intrinsics_s, intrinsics_t, target_image_shape,padding_mode='zeros'):
    """
    remap source image to a new intrinsics
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        intrinsics_s: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_t: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image remapped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    print(img.shape,intrinsics_s.shape,intrinsics_t.shape)
    global src_pixel_coords
    check_sizes(img, 'img', 'B3HW')
    check_sizes(intrinsics_s, 'intrinsics', 'B33')
    check_sizes(intrinsics_t, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()
    intrinsics_trans = intrinsics_s @ intrinsics_t.inverse()
    print(intrinsics_trans)
    if src_pixel_coords == None:
        src_pixel_coords = get_pixed_coords(intrinsics_trans,target_image_shape,(img_width,img_height)) # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points
