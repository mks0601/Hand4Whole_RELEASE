import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import pickle
import cv2
import torch
from glob import glob
from pycocotools.coco import COCO
from utils.human_models import smpl_x, smpl, mano, flame
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
from utils.transforms import rigid_align
from utils.vis import vis_keypoints, vis_mesh, save_obj

class AGORA(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'AGORA', 'data')
        self.resolution = (2160, 3840) # height, width. one of (720, 1280) and (2160, 3840)
        self.test_set = 'test' # val, test
        
        # AGORA joint set
        self.joint_set = {
                            'smplx_orig': { \
                                    'joints_name': ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',  # body
                                'Jaw', 'L_Eye_SMPLH', 'R_Eye_SMPLH',  # SMPLH
                                'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',  # fingers
                                'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',  # fingers
                                'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear',  # face in body
                                'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # feet
                                'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4',  # finger tips
                                'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4', # finger tips
                                *['Face_' + str(i) for i in range(5,56)]) # face
                                },

                            'hand': { \
                                    'joint_num': 21,
                                    'joints_name': ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_4', 'Middle_4', 'Ring_4', 'Pinky_4'),
                                    'flip_pairs': ()
                                    },

                            'face': {\
                                    'joints_name': ['Neck'] + ['Face_' + str(i) for i in range(5,56)],
                                    'joint_to_flame': (0, -1, -1, -1, -1, # no backheads and eyeballs keypoints
                                                    0, 1, 2, 3, 4, # right eyebrow
                                                    5, 6, 7, 8, 9, # left eyebrow
                                                    10, 11, 12, 13, # nose
                                                    14, 15, 16, 17, 18, # below nose
                                                    19, 20, 21, 22, 23, 24, # right eye
                                                    25, 26, 27, 28, 29, 30, # left eye
                                                    31, # right lip
                                                    32, 33, 34, 35, 36, # top lip
                                                    37, # left lip
                                                    38, 39, 40, 41, 42, # down lip
                                                    43, 44, 45, 46, 47, 48, 49, 50, # inside of lip
                                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 # no face contour keypoints
                                                    )
                                    },
                            'body': {\
                                    'joint_num': 45,
                                    'joints_name': ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4', 'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4'),
                                    'flip_pairs': ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23), (25,26), (27,28), (29,32), (30,33), (31,34), (35,40), (36,41), (37,42), (38,43), (39,44) )
                                    }
                            }

        self.joint_set['hand']['root_joint_idx'] = self.joint_set['hand']['joints_name'].index('Wrist')
        self.joint_set['hand']['orig_to_lhand'] = [self.joint_set['smplx_orig']['joints_name'].index('L_' + name) for name in self.joint_set['hand']['joints_name']]
        self.joint_set['hand']['orig_to_rhand'] = [self.joint_set['smplx_orig']['joints_name'].index('R_' + name) for name in self.joint_set['hand']['joints_name']]

        self.joint_set['face']['root_joint_idx'] = self.joint_set['face']['joints_name'].index('Neck')
        self.joint_set['face']['orig_to_face'] = [self.joint_set['smplx_orig']['joints_name'].index(name) for name in self.joint_set['face']['joints_name']]

        self.joint_set['body']['root_joint_idx'] = self.joint_set['body']['joints_name'].index('Pelvis')
        self.datalist = self.load_data()

    def load_data(self):
        datalist = []

        if self.data_split == 'train' or (self.data_split == 'test' and self.test_set == 'val'):
            if self.data_split == 'train':
                db = COCO(osp.join(self.data_path, 'AGORA_train.json'))
            else:
                db = COCO(osp.join(self.data_path, 'AGORA_validation.json'))
            
            for aid in db.anns.keys():
                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                if not ann['is_valid']:
                    continue

                if self.resolution == (720, 1280):
                    img_shape = (img['height'], img['width'])
                    img_path = osp.join(self.data_path, img['file_name_1280x720'])
                
                    if cfg.parts == 'body':
                        joints_2d_path = osp.join(self.data_path, ann['smpl_joints_2d_path'])
                        joints_3d_path = osp.join(self.data_path, ann['smpl_joints_3d_path'])
                        verts_path = osp.join(self.data_path, ann['smpl_verts_path'])
                        param_path = osp.join(self.data_path, ann['smpl_param_path'])

                        # convert to current resolution
                        bbox = np.array(ann['bbox']).reshape(4)
                        bbox[:,0] = bbox[:,0] / 3840 * 1280
                        bbox[:,1] = bbox[:,1] / 2160 * 720
                        bbox = bbox.reshape(4)
                        bbox = process_bbox(bbox, img_shape[1], img_shape[0])
                        if bbox is None:
                            continue

                        data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'joints_2d_path': joints_2d_path, 'joints_3d_path': joints_3d_path, 'verts_path': verts_path, 'param_path': param_path}
                        datalist.append(data_dict)

                    elif cfg.parts == 'hand':
                        joints_2d_path = osp.join(self.data_path, ann['smplx_joints_2d_path'])
                        joints_3d_path = osp.join(self.data_path, ann['smplx_joints_3d_path'])
                        verts_path = osp.join(self.data_path, ann['smplx_verts_path'])
                        _param_path = osp.join(self.data_path, ann['smplx_param_path'])

                        for hand_type in ('left', 'right'):
                            # convert to current resolution
                            bbox = np.array(ann[hand_type[0] + 'hand_bbox']).reshape(4)
                            bbox[:,0] = bbox[:,0] / 3840 * 1280
                            bbox[:,1] = bbox[:,1] / 2160 * 720
                            bbox = bbox.reshape(4)
                            bbox = process_bbox(bbox, img_shape[1], img_shape[0])
                            if bbox is None:
                                continue
                            data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'joints_2d_path': joints_2d_path, 'joints_3d_path': joints_3d_path, 'verts_path': verts_path, 'param_path': param_path, 'hand_type': hand_type}
                            datalist.append(data_dict)

                    elif cfg.parts == 'face':
                        joints_2d_path = osp.join(self.data_path, ann['smplx_joints_2d_path'])
                        joints_3d_path = osp.join(self.data_path, ann['smplx_joints_3d_path'])
                        verts_path = osp.join(self.data_path, ann['smplx_verts_path'])
                        param_path = osp.join(self.data_path, ann['smplx_param_path'])

                        # convert to current resolution
                        bbox = np.array(ann['face_bbox']).reshape(4)
                        bbox[:,0] = bbox[:,0] / 3840 * 1280
                        bbox[:,1] = bbox[:,1] / 2160 * 720
                        bbox = bbox.reshape(4)
                        bbox = process_bbox(bbox, img_shape[1], img_shape[0])
                        if bbox is None:
                            continue

                        data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'joints_2d_path': joints_2d_path, 'joints_3d_path': joints_3d_path, 'verts_path': verts_path, 'param_path': param_path}
                        datalist.append(data_dict)

                elif self.resolution == (2160, 3840): # use cropped and resized images. loading 4K images in pytorch dataloader takes too much time...
                    img_path = osp.join(self.data_path, '3840x2160', img['file_name_3840x2160'].split('/')[-2] + '_crop', img['file_name_3840x2160'].split('/')[-1][:-4] + '_ann_id_' + str(aid) + '.png')
                    json_path = osp.join(self.data_path, '3840x2160', img['file_name_3840x2160'].split('/')[-2] + '_crop', img['file_name_3840x2160'].split('/')[-1][:-4] + '_ann_id_' + str(aid) + '.json')
                    if not osp.isfile(json_path):
                        continue
                    with open(json_path) as f:
                        crop_resize_info = json.load(f)
                        img2bb_trans_from_orig = np.array(crop_resize_info['img2bb_trans'], dtype=np.float32)
                        resized_height, resized_width = crop_resize_info['resized_height'], crop_resize_info['resized_width']
                    img_shape = (resized_height, resized_width)

                    if cfg.parts == 'body':
                        joints_2d_path = osp.join(self.data_path, ann['smpl_joints_2d_path'])
                        joints_3d_path = osp.join(self.data_path, ann['smpl_joints_3d_path'])
                        verts_path = osp.join(self.data_path, ann['smpl_verts_path'])
                        param_path = osp.join(self.data_path, ann['smpl_param_path'])
                        bbox = np.array([0, 0, resized_width, resized_height], dtype=np.float32)
                        data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'img2bb_trans_from_orig': img2bb_trans_from_orig, 'joints_2d_path': joints_2d_path, 'joints_3d_path': joints_3d_path, 'verts_path': verts_path, 'param_path': param_path}
                        datalist.append(data_dict)

                    elif cfg.parts == 'hand':
                        joints_2d_path = osp.join(self.data_path, ann['smplx_joints_2d_path'])
                        joints_3d_path = osp.join(self.data_path, ann['smplx_joints_3d_path'])
                        verts_path = osp.join(self.data_path, ann['smplx_verts_path'])
                        param_path = osp.join(self.data_path, ann['smplx_param_path'])

                        for hand_type in ('left', 'right'):
                            # convert from original (3840,2160) to cropped space
                            bbox = np.array(ann[hand_type[0] + 'hand_bbox']).reshape(2,2)
                            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:,:1])),1)
                            bbox = np.dot(img2bb_trans_from_orig, bbox_xy1.transpose(1,0)).transpose(1,0)
                            bbox = bbox.reshape(4)
                            bbox = process_bbox(bbox, img_shape[1], img_shape[0])
                            if bbox is None:
                                continue
                            data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'img2bb_trans_from_orig': img2bb_trans_from_orig, 'joints_2d_path': joints_2d_path, 'joints_3d_path': joints_3d_path, 'verts_path': verts_path, 'param_path': param_path, 'hand_type': hand_type}
                            datalist.append(data_dict)

                    elif cfg.parts == 'face':
                        joints_2d_path = osp.join(self.data_path, ann['smplx_joints_2d_path'])
                        joints_3d_path = osp.join(self.data_path, ann['smplx_joints_3d_path'])
                        verts_path = osp.join(self.data_path, ann['smplx_verts_path'])
                        param_path = osp.join(self.data_path, ann['smplx_param_path'])
                        
                        # convert from original (3840,2160) to cropped space
                        bbox = np.array(ann['face_bbox']).reshape(2,2)
                        bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:,:1])),1)
                        bbox = np.dot(img2bb_trans_from_orig, bbox_xy1.transpose(1,0)).transpose(1,0)
                        bbox = bbox.reshape(4)
                        bbox = process_bbox(bbox, img_shape[1], img_shape[0])
                        if bbox is None:
                            continue
                        data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'img2bb_trans_from_orig': img2bb_trans_from_orig, 'joints_2d_path': joints_2d_path, 'joints_3d_path': joints_3d_path, 'verts_path': verts_path, 'param_path': param_path}
                        datalist.append(data_dict)

        elif self.data_split == 'test' and self.test_set == 'test': 
            assert cfg.parts == 'body' # the evaluation server does not support MANO- and FLAME-only evaluations

            with open(osp.join(self.data_path, 'AGORA_test_bbox.json')) as f:
                bboxs = json.load(f)

            for filename in bboxs.keys():
                if self.resolution == (720, 1280):
                    img_path = osp.join(self.data_path, 'test', filename)
                    img_shape = self.resolution
                    person_num = len(bboxs[filename])
                    for pid in range(person_num):
                        # change bbox from (2160,3840) to target resoution
                        bbox = np.array(bboxs[filename][pid]['bbox']).reshape(2,2)
                        bbox[:,0] = bbox[:,0] / 3840 * 1280
                        bbox[:,1] = bbox[:,1] / 2160 * 720
                        bbox = bbox.reshape(4)
                        bbox = process_bbox(bbox, img_shape[1], img_shape[0])
                        if bbox is None:
                            continue
                        datalist.append({'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'person_idx': pid})

                elif self.resolution == (2160, 3840): # use cropped and resized images. loading 4K images in pytorch dataloader takes too much time...
                    person_num = len(bboxs[filename])
                    for pid in range(person_num):
                        img_path = osp.join(self.data_path, '3840x2160', 'test_crop', filename[:-4] + '_pid_' + str(pid) + '.png')
                        json_path = osp.join(self.data_path, '3840x2160', 'test_crop', filename[:-4] + '_pid_' + str(pid) + '.json')
                        if not osp.isfile(json_path):
                            print(json_path)
                            continue
                        with open(json_path) as f:
                            crop_resize_info = json.load(f)
                            img2bb_trans_from_orig = np.array(crop_resize_info['img2bb_trans'], dtype=np.float32)
                            resized_height, resized_width = crop_resize_info['resized_height'], crop_resize_info['resized_width']
                        img_shape = (resized_height, resized_width)
                        bbox = np.array([0, 0, resized_width, resized_height], dtype=np.float32)
                        datalist.append({'img_path': img_path, 'img_shape': img_shape, 'img2bb_trans_from_orig': img2bb_trans_from_orig, 'bbox': bbox, 'person_idx': pid})


        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
        
        # image load
        img = load_img(img_path)
        
        # affine transform
        if cfg.parts == 'hand':
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split, enforce_flip=(data['hand_type'] == 'left')) # enforce flip when left hand to make it right hand
        else:
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split) 
        img = self.transform(img.astype(np.float32))/255.

        # train mode
        if self.data_split == 'train':
            # gt load
            with open(data['joints_2d_path']) as f:
                joint_img = np.array(json.load(f)).reshape(-1,2)
                if self.resolution == (2160, 3840):
                    joint_img[:,:2] = np.dot(data['img2bb_trans_from_orig'], np.concatenate((joint_img, np.ones_like(joint_img[:,:1])),1).transpose(1,0)).transpose(1,0) # transform from original image to crop_and_resize image
                else:
                    joint_img[:,0] = joint_img[:,0] / 3840 * self.resolution[1]
                    joint_img[:,1] = joint_img[:,1] / 2160 * self.resolution[0]
            with open(data['joints_3d_path']) as f:
                joint_cam = np.array(json.load(f)).reshape(-1,3)
            with open(data['param_path'], 'rb') as f:
                param = pickle.load(f, encoding='latin1')

            # body part
            if cfg.parts == 'body':
                # coordinates
                joint_cam = joint_cam - joint_cam[self.joint_set['body']['root_joint_idx'],None,:] # root-relative
                joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1) # x, y, depth
                joint_valid = np.ones_like(joint_img[:,:1])
                joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, joint_cam, joint_valid, do_flip, img_shape, self.joint_set['body']['flip_pairs'], img2bb_trans, rot, self.joint_set['body']['joints_name'], smpl.joints_name)

                # smpl parameters
                root_pose = np.array(param['root_pose'], dtype=np.float32).reshape(-1) # rotation to world coordinate
                body_pose = np.array(param['body_pose'], dtype=np.float32).reshape(-1)
                smpl_pose = np.concatenate((root_pose, body_pose))
                shape = np.array(param['betas'], dtype=np.float32).reshape(-1)[:10] # bug?
                trans = np.array(param['translation'], dtype=np.float32).reshape(-1) # translation to world coordinate
                smpl_param = {'pose': smpl_pose, 'shape': shape, 'trans': trans}
                cam_param = {'focal': cfg.focal, 'princpt': cfg.princpt} # put random camera paraemter as we do not use coordinates from smpl parameters
                _, _, _, smpl_pose, smpl_shape, _ = process_human_model_output(smpl_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smpl')
                smpl_pose_valid = np.ones((smpl.orig_joint_num*3), dtype=np.float32)
                smpl_pose_valid[:3] = 0 # global orient of the provided parameter is a rotation to world coordinate system. I want camera coordinate system.
                
                """
                # for debug
                _tmp = joint_img.copy()
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1].copy() * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('agora_' + str(idx) + '.jpg', _img)
                """

                inputs = {'img': img}
                targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'smpl_joint_img': joint_img, 'smpl_joint_cam': joint_cam, 'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape}
                meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smpl_joint_valid': np.zeros_like(joint_valid), 'smpl_joint_trunc': np.zeros_like(joint_trunc), 'smpl_pose_valid': smpl_pose_valid, 'smpl_shape_valid': float(True), 'is_3D': float(True)}
                return inputs, targets, meta_info

            # hand part
            elif cfg.parts == 'hand':
                hand_type = data['hand_type']
                if hand_type == 'left':
                    joint_img = joint_img[self.joint_set['hand']['orig_to_lhand'],:]
                    joint_cam = joint_cam[self.joint_set['hand']['orig_to_lhand'],:]
                    root_pose = np.array(param['global_orient'], dtype=np.float32).reshape(-1) # rotation to world coordinate
                    lhand_pose = np.array(param['left_hand_pose'], dtype=np.float32).reshape(-1)
                    mano_pose = np.concatenate((root_pose, lhand_pose))
                else:
                    joint_img = joint_img[self.joint_set['hand']['orig_to_rhand'],:]
                    joint_cam = joint_cam[self.joint_set['hand']['orig_to_rhand'],:]
                    root_pose = np.array(param['global_orient'], dtype=np.float32).reshape(-1) # rotation to world coordinate
                    rhand_pose = np.array(param['right_hand_pose'], dtype=np.float32).reshape(-1)
                    mano_pose = np.concatenate((root_pose, rhand_pose))

                # coordinates
                joint_cam = joint_cam - joint_cam[self.joint_set['hand']['root_joint_idx'],None,:] # root-relative
                joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1) # x, y, depth
                joint_valid = np.ones_like(joint_img[:,:1])
                joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, joint_cam, joint_valid, do_flip, img_shape, self.joint_set['hand']['flip_pairs'], img2bb_trans, rot, self.joint_set['hand']['joints_name'], mano.joints_name)
                
                """
                # for debug
                _tmp = joint_img.copy()
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1].copy() * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('agora_' + str(idx) + '_' + hand_type + '.jpg', _img)
                """

                # mano parameters
                shape = np.array(param['betas'], dtype=np.float32).reshape(-1)[:10] # bug?
                trans = np.array(param['transl'], dtype=np.float32).reshape(-1) # translation to world coordinate
                mano_param = {'pose': mano_pose, 'shape': shape, 'trans': trans, 'hand_type': hand_type}
                cam_param = {'focal': cfg.focal, 'princpt': cfg.princpt} # put random camera paraemter as we do not use coordinates from smplx parameters
                _, _, _, mano_pose, mano_shape, _ = process_human_model_output(mano_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'mano')
                mano_pose_valid = np.ones((mano.orig_joint_num*3), dtype=np.float32)
                mano_pose_valid[:3] = 0 # global orient of the provided parameter is a rotation to world coordinate system of whole body. I want camera coordinate system of a hand.

                inputs = {'img': img}
                targets = {'joint_img': joint_img, 'mano_joint_img': joint_img, 'joint_cam': joint_cam, 'mano_joint_cam': joint_cam, 'mano_pose': mano_pose, 'mano_shape': mano_shape}
                meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'mano_joint_trunc': np.zeros_like(joint_trunc), 'mano_pose_valid': np.zeros_like(mano_pose_valid), 'is_valid_mano_fit': float(True), 'is_3D': float(True)}
                return inputs, targets, meta_info

            # face part
            elif cfg.parts == 'face':
                # change keypoint set to that of flame
                joint_img = joint_img[self.joint_set['face']['orig_to_face'],:]
                flame_joint_img = np.zeros((flame.joint_num,3), dtype=np.float32)
                flame_joint_valid = np.zeros((flame.joint_num,1), dtype=np.float32)
                for j in range(flame.joint_num):
                    if self.joint_set['face']['joint_to_flame'][j] == -1:
                        continue
                    flame_joint_img[j] = joint_img[self.joint_set['face']['joint_to_flame'][j]]
                    flame_joint_valid[j] = joint_valid[self.joint_set['face']['joint_to_flame'][j]]
                joint_img = flame_joint_img
                joint_valid = flame_joint_valid

                # coordinates
                joint_cam = joint_cam - joint_cam[self.joint_set['face']['root_joint_idx'],None,:] # root-relative
                joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1) # x, y, depth
                joint_valid = np.ones_like(joint_img[:,:1])
                joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, joint_cam, joint_valid, do_flip, img_shape, flame.flip_pairs, img2bb_trans, rot, flame.joints_name, flame.joints_name)

                # flame parameters
                root_pose = np.array(param['global_orient'], dtype=np.float32).reshape(-1) # rotation to world coordinate
                shape = np.array(param['betas'], dtype=np.float32).reshape(-1)[:10] # bug?
                jaw_pose = np.array(param['jaw_pose'], dtype=np.float32).reshape(-1)
                expr = np.array(param['expression'], dtype=np.float32).reshape(-1)
                trans = np.array(param['transl'], dtype=np.float32).reshape(-1) # translation to world coordinate

                cam_param = {'focal': cfg.focal, 'princpt': cfg.princpt} # put random camera paraemter as we do not use coordinates from smplx parameters
                flame_param = {'root_pose': root_pose, 'jaw_pose': jaw_pose, 'shape': shape, 'expr': expr, 'trans': trans}
                _, _, _, flame_root_pose, flame_jaw_pose, flame_shape, flame_expr, _, _ = process_human_model_output(flame_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'flame')
                flame_root_pose_valid = float(False)
                 
                inputs = {'img': img}
                targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'flame_joint_cam': joint_cam, 'flame_root_pose': flame_root_pose, 'flame_jaw_pose': flame_jaw_pose, 'flame_shape': flame_shape, 'flame_expr': flame_expr}
                meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'flame_root_pose_valid': flame_root_pose_valid, 'is_3D': float(True), 'is_valid_flame_fit': float(True)}
                return inputs, targets, meta_info

        # test mode
        else:
            # load crop and resize information (for the 4K setting)
            if self.resolution == (2160,3840):
                img2bb_trans = np.dot(
                                    np.concatenate((img2bb_trans,
                                                    np.array([0,0,1], dtype=np.float32).reshape(1,3))),
                                    np.concatenate((data['img2bb_trans_from_orig'],
                                                    np.array([0,0,1], dtype=np.float32).reshape(1,3)))
                                    )
                bb2img_trans = np.linalg.inv(img2bb_trans)[:2,:]

            
            if self.test_set == 'val':
                with open(data['verts_path']) as f:
                    verts = np.array(json.load(f)).reshape(-1,3)

                if cfg.parts == 'body':
                    inputs = {'img': img}
                    targets = {'smpl_mesh_cam': verts}
                    meta_info = {'bb2img_trans': bb2img_trans}
                    return inputs, targets, meta_info

                elif cfg.parts == 'hand':
                    mano_mesh_cam = verts[smpl_x.hand_vertex_idx[data['hand_type'] + '_hand'],:]
                    if data['hand_type'] == 'left':
                        mano_mesh_cam[:,0] *= -1 # flip left hand to right hand
                    inputs = {'img': img}
                    targets = {'mano_mesh_cam': mano_mesh_cam}
                    meta_info = {}
                    return inputs, targets, meta_info

                elif cfg.parts == 'face':
                    flame_mesh_cam = verts[smpl_x.face_vertex_idx,:]
                    inputs = {'img': img}
                    targets = {'flame_mesh_cam': flame_mesh_cam}
                    meta_info = {}
                    return inputs, targets, meta_info
            else:
                inputs = {'img': img}
                targets = {'smpl_mesh_cam': np.zeros((smpl.vertex_num, 3), dtype=np.float32)} # dummy vertex
                meta_info = {'bb2img_trans': bb2img_trans}
                return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'pa_mpvpe': [], 'mpvpe': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]

            if cfg.parts == 'body':
                mesh_gt = out['smpl_mesh_cam_target']
                mesh_out = out['smpl_mesh_cam']
               
                mesh_out_align = mesh_out - np.dot(smpl.joint_regressor, mesh_out)[smpl.root_joint_idx,None,:] + np.dot(smpl.joint_regressor, mesh_gt)[smpl.root_joint_idx,None,:]
                eval_result['mpvpe'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt)**2,1)).mean() * 1000)
                mesh_out_align = rigid_align(mesh_out, mesh_gt)
                eval_result['pa_mpvpe'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt)**2,1)).mean() * 1000)
            
            elif cfg.parts == 'hand':
                mesh_gt = out['mano_mesh_cam_target']
                mesh_out = out['mano_mesh_cam']
                hand_type = annot['hand_type']
                if hand_type == 'left':
                    mesh_out[:,0] = mesh_out[:,0] * -1 # flip back to the left hand

                mesh_out_align = mesh_out - np.dot(mano.joint_regressor, mesh_out)[mano.root_joint_idx,None,:] + np.dot(mano.joint_regressor, mesh_gt)[mano.root_joint_idx,None,:]
                eval_result['mpvpe'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt)**2,1)).mean() * 1000)
                mesh_out_align = rigid_align(mesh_out, mesh_gt)
                eval_result['pa_mpvpe'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt)**2,1)).mean() * 1000)

            elif cfg.parts == 'face':
                mesh_gt = out['flame_mesh_cam_target']
                mesh_out = out['flame_mesh_cam']

                mesh_out_align = mesh_out - np.dot(flame.layer.J_regressor, mesh_out)[flame.root_joint_idx,None,:] + np.dot(flame.layer.J_regressor, mesh_gt)[flame.root_joint_idx,None,:]
                eval_result['mpvpe'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt)**2,1)).mean() * 1000)
                mesh_out_align = rigid_align(mesh_out, mesh_gt)
                eval_result['pa_mpvpe'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt)**2,1)).mean() * 1000)
           
            vis = False
            if vis:
                img = (out['img'].transpose(1,2,0)[:,:,::-1] * 255).copy()
                cv2.imwrite(str(cur_sample_idx + n) + '.jpg', img)
                
                if cfg.parts == 'body':
                    vis_mesh_out = out['smpl_mesh_cam']
                    vis_mesh_out = vis_mesh_out - np.dot(smpl.joint_regressor, vis_mesh_out)[smpl.root_joint_idx,None,:]
                    vis_mesh_gt = out['smpl_mesh_cam_target']
                    vis_mesh_gt = vis_mesh_gt - np.dot(smpl.joint_regressor, vis_mesh_gt)[smpl.root_joint_idx,None,:] 
                    save_obj(vis_mesh_out, smpl.face, str(cur_sample_idx + n) + '.obj')
                    save_obj(vis_mesh_gt, smpl.face, str(cur_sample_idx + n) + '_gt.obj')

                elif cfg.parts == 'hand':
                    vis_mesh_out = out['mano_mesh_cam']
                    vis_mesh_out = vis_mesh_out - np.dot(mano.joint_regressor, vis_mesh_out)[mano.root_joint_idx,None,:]
                    vis_mesh_gt = out['mano_mesh_cam_target']
                    vis_mesh_gt = vis_mesh_gt - np.dot(mano.joint_regressor, vis_mesh_gt)[mano.root_joint_idx,None,:] 
                    save_obj(vis_mesh_out, mano.face[hand_type], str(cur_sample_idx + n) + '_' + hand_type + '_.obj')
                    save_obj(vis_mesh_gt, mano.face[hand_type], str(cur_sample_idx + n) + '_' + hand_type + '_gt.obj')

                elif cfg.parts == 'face':
                    vis_mesh_out = out['flame_mesh_cam']
                    vis_mesh_out = vis_mesh_out - np.dot(flame.layer.J_regressor, vis_mesh_out)[flame.root_joint_idx,None,:]
                    vis_mesh_gt = out['flame_mesh_cam_target']
                    vis_mesh_gt = vis_mesh_gt - np.dot(flame.layer.J_regressor, vis_mesh_gt)[flame.root_joint_idx,None,:] 
                    save_obj(vis_mesh_out, flame.face, str(cur_sample_idx + n) + '.obj')
                    save_obj(vis_mesh_gt, flame.face, str(cur_sample_idx + n) + '_gt.obj')

            
            # official evaluation server only support body evaluation
            if cfg.parts != 'body':
                continue

            # save results for the official evaluation codes/server
            save_name = annot['img_path'].split('/')[-1][:-4]
            if self.data_split == 'test' and self.test_set == 'test':
                if self.resolution == (2160,3840):
                    save_name = save_name.split('_pid')[0]
            elif self.data_split == 'test' and self.test_set == 'val':
                if self.resolution == (2160,3840):
                    save_name = save_name.split('_ann_id')[0]
                else:
                    save_name = save_name.split('_1280x720')[0]
            if 'person_idx' in annot:
                person_idx = annot['person_idx']
            else:
                exist_result_path = glob(osp.join(cfg.result_dir, 'AGORA', save_name + '*'))
                if len(exist_result_path) == 0:
                    person_idx = 0
                else:
                    last_person_idx = max([int(name.split('personId_')[1].split('.pkl')[0]) for name in exist_result_path])
                    person_idx = last_person_idx + 1
            save_name += '_personId_' + str(person_idx) + '.pkl'

            joint_proj = out['smpl_joint_proj']
            joint_proj[:,0] = joint_proj[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            joint_proj[:,1] = joint_proj[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            joint_proj = np.concatenate((joint_proj, np.ones_like(joint_proj[:,:1])),1)
            joint_proj = np.dot(out['bb2img_trans'], joint_proj.transpose(1,0)).transpose(1,0)
            joint_proj[:,0] = joint_proj[:,0] / self.resolution[1] * 3840 # restore to original resolution
            joint_proj[:,1] = joint_proj[:,1] / self.resolution[0] * 2160 # restore to original resolution
            save_dict = {'params': 
                                {'translation': out['cam_trans'].reshape(1,-1),
                                'root_pose': out['smpl_pose'][:3].reshape(1,-1),
                                'body_pose': out['smpl_pose'][3:].reshape(1,-1),
                                'betas': out['smpl_shape'].reshape(1,-1)},
                        'joints': joint_proj.reshape(1,-1,2),
                        }
            with open(osp.join(cfg.result_dir, 'AGORA', save_name), 'wb') as f:
                pickle.dump(save_dict, f)
            
            """
            # for debug
            img_path = annot['img_path']
            img_path = osp.join(self.data_path, '3840x2160', 'test', img_path.split('/')[-1].split('_')[0] + '.png')
            img = cv2.imread(img_path)
            img = vis_keypoints(img.copy(), joint_proj)
            cv2.imwrite(img_path.split('/')[-1], img)
            """

        return eval_result

    def print_eval_result(self, eval_result):
        
        print('AGORA test results are dumped at: ' + osp.join(cfg.result_dir, 'AGORA'))
        
        if self.data_split == 'test' and self.test_set == 'test': # do not print. just submit the results to the official evaluation server
            return

        print('PA MPVPE: %.2f mm' % np.mean(eval_result['pa_mpvpe']))
        print('MPVPE: %.2f mm' % np.mean(eval_result['mpvpe']))


