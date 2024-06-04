import json
import torch
import numpy as np
import os.path as osp
from glob import glob
from tqdm import tqdm
import cv2
import pickle
import os
import pathlib
import argparse

def get_bbox(joint_img, joint_valid):
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, dest='dataset_path')
    args = parser.parse_args()
    assert args.dataset_path, "Please set dataset_path"
    return args

args = parse_args()
dataset_path = args.dataset_path

image_id = 0
ann_id = 0
gt_joints_2d_path = './gt_joints_2d'
gt_joints_3d_path = './gt_joints_3d'
gt_verts_path = './gt_verts'

pathlib.Path(osp.join(dataset_path, gt_joints_2d_path, 'smpl')).mkdir(parents=True, exist_ok=True)
pathlib.Path(osp.join(dataset_path, gt_joints_3d_path, 'smpl')).mkdir(parents=True, exist_ok=True)
pathlib.Path(osp.join(dataset_path, gt_verts_path, 'smpl')).mkdir(parents=True, exist_ok=True)

for split in ('train', 'validation'):
    images = []
    annotations = []
    data_path_list = glob(osp.join(dataset_path, split + '_SMPL', 'SMPL', '*.pkl'))
    data_path_list = sorted(data_path_list)

    for data_path in tqdm(data_path_list):
        with open(data_path, 'rb') as f:
            data_smpl = pickle.load(f, encoding='latin1')
            data_smpl = {k: list(v) for k,v in data_smpl.items()}

        if split == 'train':
            img_folder_name = data_path.split('/')[-1].split('_withjv')[0] # e.g., train_0
        else:
            img_folder_name = 'validation'
        img_num = len(data_smpl['imgPath'])
        
        for i in range(img_num):
            img_dict = {}
            img_dict['id'] = image_id
            img_dict['file_name_3840x2160'] = osp.join('3840x2160', img_folder_name, data_smpl['imgPath'][i])
            img_dict['file_name_1280x720'] = osp.join('1280x720', img_folder_name, data_smpl['imgPath'][i][:-4] + '_1280x720.png')
            images.append(img_dict)

            person_num = len(data_smpl['gt_path_smpl'][i])
            for j in range(person_num):
                ann_dict = {}
                ann_dict['id'] = ann_id
                ann_dict['image_id'] = image_id
                ann_dict['smpl_joints_2d_path'] = osp.join(gt_joints_2d_path, 'smpl', str(ann_id) + '.json')
                ann_dict['smpl_joints_3d_path'] = osp.join(gt_joints_3d_path, 'smpl', str(ann_id) + '.json')
                ann_dict['smpl_verts_path'] = osp.join(gt_verts_path, 'smpl', str(ann_id) + '.json')
                ann_dict['smpl_param_path'] = data_smpl['gt_path_smpl'][i][j][:-4] + '.pkl'
                ann_dict['gender'] = data_smpl['gender'][i][j]
                ann_dict['kid'] = data_smpl['kid'][i][j]
                ann_dict['occlusion'] = data_smpl['occlusion'][i][j]
                ann_dict['is_valid'] = data_smpl['isValid'][i][j]
                ann_dict['age'] = data_smpl['age'][i][j]
                ann_dict['ethnicity'] = data_smpl['ethnicity'][i][j]
                
                # bbox
                joints_2d = np.array(data_smpl['gt_joints_2d'][i][j]).reshape(-1,2)
                bbox = get_bbox(joints_2d, np.ones_like(joints_2d[:,0])).reshape(4)
                ann_dict['bbox'] = bbox.tolist()
                
                # save smpl gts
                joints_2d = np.array(data_smpl['gt_joints_2d'][i][j]).reshape(-1,2)
                with open(osp.join(dataset_path, gt_joints_2d_path, 'smpl', str(ann_id) + '.json'), 'w') as f:
                    json.dump(joints_2d.tolist(), f)
                joints_3d = np.array(data_smpl['gt_joints_3d'][i][j]).reshape(-1,3)
                with open(osp.join(dataset_path, gt_joints_3d_path, 'smpl', str(ann_id) + '.json'), 'w') as f:
                    json.dump(joints_3d.tolist(), f)
                verts = np.array(data_smpl['gt_verts'][i][j]).reshape(-1,3)
                with open(osp.join(dataset_path, gt_verts_path, 'smpl', str(ann_id) + '.json'), 'w') as f:
                    json.dump(verts.tolist(), f)

                ann_id += 1
            image_id += 1

    with open(osp.join(dataset_path, 'AGORA_' + split + '_SMPL.json'), 'w') as f:
        json.dump({'images': images, 'annotations': annotations}, f)
        

    
