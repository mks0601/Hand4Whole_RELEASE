import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from glob import glob
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import mano
from utils.preprocessing import load_img, get_bbox, process_bbox, augmentation, process_db_coord, process_human_model_output
from utils.transforms import world2cam, cam2pixel, rigid_align
from utils.vis import vis_keypoints, vis_mesh, save_obj

class InterHand26M(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('..', 'data', 'InterHand26M', 'images')
        self.annot_path = osp.join('..', 'data', 'InterHand26M', 'annotations')

        # IH26M joint set
        self.joint_set = {'hand': \
                            {'joint_num': 21, # single hand
                            'joints_name': ('Thumb_4', 'Thumb_3', 'Thumb_2', 'Thumb_1', 'Index_4', 'Index_3', 'Index_2', 'Index_1', 'Middle_4', 'Middle_3', 'Middle_2', 'Middle_1', 'Ring_4', 'Ring_3', 'Ring_2', 'Ring_1', 'Pinky_4', 'Pinky_3', 'Pinky_2', 'Pinky_1', 'Wrist'),
                            'flip_pairs': ()
                            }
                        }
        self.joint_set['hand']['joint_type'] = {'right': np.arange(0,self.joint_set['hand']['joint_num']), 'left': np.arange(self.joint_set['hand']['joint_num'],self.joint_set['hand']['joint_num']*2)}
        self.joint_set['hand']['root_joint_idx'] = self.joint_set['hand']['joints_name'].index('Wrist')
        self.datalist = self.load_data()
        
    def load_data(self):
        # load annotation
        db = COCO(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_data.json'))
        with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_joint_3d.json')) as f:
            joints = json.load(f)
        with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_MANO_NeuralAnnot.json')) as f:
            mano_params = json.load(f)
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.img_path, self.data_split, img['file_name'])

            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']

            # too many views :( camera filtering in the training stage
            if self.data_split == 'train':
                cam_folder_list = glob(osp.join(self.img_path, 'train', 'Capture' + str(capture_id), seq_name, '*'))
                cam_folder_list = [path.split('/')[-1] for path in cam_folder_list]
                cam_folder_list = sorted(cam_folder_list)
                if len(cam_folder_list) < 10:
                    step_size = 1
                else:
                    step_size = len(cam_folder_list) // 10
                cam_folder_list = cam_folder_list[::step_size] # reduce the number of cameras by 10 times
                if 'cam' + str(cam) not in cam_folder_list:
                    continue

            # camera parameters
            t, R = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32).reshape(3), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32).reshape(3,3)
            t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32).reshape(2), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32).reshape(2)
            cam_param = {'R': R, 't': t, 'focal': focal, 'princpt': princpt}
           
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(-1,1)
            joint_valid[self.joint_set['hand']['joint_type']['right']] *= joint_valid[self.joint_set['hand']['root_joint_idx']]
            joint_valid[self.joint_set['hand']['joint_type']['left']] *= joint_valid[self.joint_set['hand']['joint_num'] + self.joint_set['hand']['root_joint_idx']]

            # joint coordinates
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32).reshape(-1,3)
            joint_cam = world2cam(joint_world, R, t)
            joint_cam[np.tile(joint_valid==0, (1,3))] = 1. # prevent zero division error
            joint_img = cam2pixel(joint_cam, focal, princpt)
            
            # add right and left hand
            if ann['hand_type'] == 'right':
                hand_type_list = ('right',)
            elif ann['hand_type'] == 'left':
                hand_type_list = ('left',)
            else:
                hand_type_list = ('right','left')
            for hand_type in hand_type_list:
                
                if np.sum(joint_valid[self.joint_set['hand']['joint_type'][hand_type]]) == 0:
                    continue
                
                # bbox 
                bbox = get_bbox(joint_img[self.joint_set['hand']['joint_type'][hand_type],:2], joint_valid[self.joint_set['hand']['joint_type'][hand_type],0], extend_ratio=1.5)
                bbox = process_bbox(bbox, img_width, img_height)
                if bbox is None:
                    continue

                # mano parameters
                try:
                    mano_param = mano_params[str(capture_id)][str(frame_idx)][hand_type]
                    if mano_param is not None:
                        mano_param['hand_type'] = hand_type
                except KeyError:
                    mano_param = None
                
                datalist.append({
                    'img_path': img_path,
                    'img_shape': (img_height, img_width),
                    'bbox': bbox,
                    'joint_img': joint_img[self.joint_set['hand']['joint_type'][hand_type],:],
                    'joint_cam': joint_cam[self.joint_set['hand']['joint_type'][hand_type],:],
                    'joint_valid': joint_valid[self.joint_set['hand']['joint_type'][hand_type],:],
                    'cam_param': cam_param,
                    'mano_param': mano_param,
                    'hand_type': hand_type})
                    # 'orig_hand_type': ann['hand_type']}) # causes memory leak..

        return datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, hand_type = data['img_path'], data['img_shape'], data['bbox'], data['hand_type']
        data['cam_param']['t'] /= 1000 # milimeter to meter

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split, enforce_flip=(hand_type=='left')) # enforce flip when left hand to make it right hand
        img = self.transform(img.astype(np.float32))/255.

        if self.data_split == 'train':
            # ih26m hand gt
            joint_cam = data['joint_cam']
            joint_cam = (joint_cam - joint_cam[self.joint_set['hand']['root_joint_idx'],None,:]) / 1000 # root-relative. milimeter to meter.
            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1)
            joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, joint_cam, data['joint_valid'], do_flip, img_shape, self.joint_set['hand']['flip_pairs'], img2bb_trans, rot, self.joint_set['hand']['joints_name'], mano.joints_name)

            # mano coordinates
            mano_param = data['mano_param']
            if mano_param is not None:
                mano_joint_img, mano_joint_cam, mano_joint_trunc, mano_pose, mano_shape, mano_mesh_cam_orig = process_human_model_output(mano_param, data['cam_param'], do_flip, img_shape, img2bb_trans, rot, 'mano')
                mano_joint_valid = np.ones((mano.joint_num,1), dtype=np.float32)
                mano_pose_valid = np.ones((mano.orig_joint_num*3), dtype=np.float32)
                mano_shape_valid = float(True)

                """
                # for debug
                _tmp = joint_img.copy()
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[1] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[0] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('ih26m_' + str(idx) + '_' + hand_type + '.jpg', _img)

                # for debug
                _tmp = mano_joint_img.copy()
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[1] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[0] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('ih26m_' + str(idx) + hand_type + '_mano.jpg', _img)
                """
            else:
                # dummy values
                mano_joint_img = np.zeros((mano.joint_num,3), dtype=np.float32)
                mano_joint_cam = np.zeros((mano.joint_num,3), dtype=np.float32)
                mano_joint_trunc = np.zeros((mano.joint_num,1), dtype=np.float32)
                mano_pose = np.zeros((mano.orig_joint_num*3), dtype=np.float32) 
                mano_shape = np.zeros((mano.shape_param_dim), dtype=np.float32)
                mano_joint_valid = np.zeros((mano.joint_num,1), dtype=np.float32)
                mano_pose_valid = np.zeros((mano.orig_joint_num*3), dtype=np.float32)
                mano_shape_valid = float(False)

            inputs = {'img': img}
            targets = {'joint_img': joint_img, 'mano_joint_img': mano_joint_img, 'joint_cam': joint_cam, 'mano_joint_cam': mano_joint_cam, 'mano_pose': mano_pose, 'mano_shape': mano_shape}
            meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'mano_joint_trunc': mano_joint_trunc, 'mano_joint_valid': mano_joint_valid, 'mano_pose_valid': mano_pose_valid, 'mano_shape_valid': mano_shape_valid, 'is_3D': float(True)}
        else:
            inputs = {'img': img}
            targets = {}
            meta_info = {}

        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        return {}
    
    def print_eval_result(self, eval_result):
        return
