'''
* Copyright (c) 2019 Carnegie Mellon University, Author <xiangwew@andrew.cmu.edu> <basti@andrew.cmu.edu>
*
* Not licensed for commercial use. For research and evaluation only.
*
'''

import argparse
#configuration
def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--tag', dest='tag', default='ntsd_', help='tag')
    parser.add_argument('--mode', dest='mode', default='train_random', help='mode')
    parser.add_argument('--motion_ax', dest='motion_ax', default='0 0 1 0 0 0', help='motion_ax')
    parser.add_argument('--test_motion_ax', dest='test_motion_ax', default='0 0 0 0 1 0', help='test_motion_ax')
    parser.add_argument('--images_train', dest='images_train', default='dataset/ntsd/ntsd.image.train', help='image_train')
    parser.add_argument('--poses_train', dest='poses_train', default='dataset/ntsd/ntsd.pose.train', help='pose_train')
    parser.add_argument('--images_target', dest='images_target', default='dataset/ntsd/ntsd.image.target', help='image_target')
    parser.add_argument('--poses_target', dest='poses_target', default='dataset/ntsd/ntsd.pose.target', help='pose_target')
    parser.add_argument('--feature_model', dest='feature_model', default='result/saved_model/feature_ntsd_100.pt', help='feature_model')
    parser.add_argument('--epoch', dest='epoch', type=int,default=100, help='epoch')
    parser.add_argument('--gpu', dest='gpu',type =int, default=1, help='gpu')

    '''
    parser.add_argument('--optimizer', dest='optimization_method', default='sgd', help='optimization method')
    parser.add_argument('--result', dest='result_path', default='result/00_predict_posang.txt', help='predict result path')
    parser.add_argument('--imagelist', dest='image_list_path', default='dataset/kitti/kitti.image.train', help='image list path')
    parser.add_argument('--motion', dest='motion_path', default='dataset/kitti//kitti.pose.train', help='motion path')
    parser.add_argument('--model', dest='model_name', default='test_att_from_0_00_simpleatt', help='model name')
    parser.add_argument('--model_load', dest='model_load', default='cmu_data_all', help='model name')
    parser.add_argument('--batch', dest='batch_size',type=int, default=1, help='batch size')
    parser.add_argument('--motion_test', dest='motion_path_test', default='dataset/kitti/kitti.pose.test', help='test motion path')
    parser.add_argument('--imagelist_test', dest='image_list_path_test', default='dataset/kitti/kitti.image.test', help='test image list path')
    parser.add_argument('--port', dest='visdom_port', default='8097', help='visdom port')
    parser.add_argument('--ip', dest='visdom_ip', default='http://127.0.0.1', help='visdom port')
    parser.add_argument('--mean_std_path', dest='mean_std_path', default='my_utils/mean_std.txt', help='visdom port')
    parser.add_argument('--check_period', dest='check_period', default=5, help='the period for visualization and testing and saving model')

    parser.add_argument('--fine_tune', dest='finetune_flag', default=False, help='fine tune')
    parser.add_argument('--use_gpu', dest='use_gpu_flag', default=True, help='use gpu')
    parser.add_argument('--coor_layer', dest='coor_layer_flag', default=True, help='coor layer')
    parser.add_argument('--pad', dest='pad_flag', default=True, help='pad')
    parser.add_argument('--attention', dest='attention_flag', default=False, help='attention')
    '''
    args = parser.parse_args()
    return args
