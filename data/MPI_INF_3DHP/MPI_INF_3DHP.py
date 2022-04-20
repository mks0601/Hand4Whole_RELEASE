import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import smpl
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
from utils.transforms import world2cam, cam2pixel
from utils.vis import vis_keypoints, vis_mesh, save_obj

class MPI_INF_3DHP(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'MPI_INF_3DHP', 'data')

        # MPI-INF-3DHP joint set
        self.joint_set = {'body': \
                            {'joint_num': 17,
                            'joints_name': ('Head_top', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Torso', 'Head'),
                            'flip_pairs': ( (2,5), (3,6), (4,7), (8,11), (9,12), (10,13) )
                            }
                        }
        self.joint_set['body']['root_joint_idx'] = self.joint_set['body']['joints_name'].index('Pelvis')

        self.datalist = self.load_data()
        
    def load_data(self):
        db = COCO(osp.join(self.data_path, 'MPI-INF-3DHP_1k.json'))
        with open(osp.join(self.data_path, 'MPI-INF-3DHP_joint_3d.json')) as f:
            joints = json.load(f)
        with open(osp.join(self.data_path, 'MPI-INF-3DHP_camera_1k.json')) as f:
            cameras = json.load(f)
        # smpl parameters load
        smpl_param_path = osp.join(self.data_path, 'MPI-INF-3DHP_SMPL_NeuralAnnot.json')
        with open(smpl_param_path,'r') as f:
            smpl_params = json.load(f)

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            subject_idx = img['subject_idx']
            seq_idx = img['seq_idx']
            frame_idx = img['frame_idx']
            img_path = osp.join(self.data_path, 'images_1k', 'S' + str(subject_idx), 'Seq' + str(seq_idx), 'imageSequence', img['file_name'])
            img_shape = (img['height'], img['width'])
            
            # frame sampling (25 frame per sec -> 25/3 frame per sec)
            if frame_idx % 3 != 0:
                continue

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject_idx)][str(seq_idx)][str(cam_idx)]
            R, t, focal, princpt = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['focal'], dtype=np.float32), np.array(cam_param['princpt'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal':focal, 'princpt':princpt}
            
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject_idx)][str(seq_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, focal, princpt)
            joint_valid = np.ones_like(joint_img[:,:1])

            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue

            # smpl parameter
            smpl_param = smpl_params[str(subject_idx)][str(seq_idx)][str(frame_idx)]
    
            datalist.append({
                'img_path': img_path,
                'img_id': image_id,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'cam_param': cam_param,
                'smpl_param': smpl_param})

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, smpl_param, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['smpl_param'], data['cam_param']

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
         
        # mi gt
        joint_cam = data['joint_cam']
        joint_cam = (joint_cam - joint_cam[self.joint_set['body']['root_joint_idx'],None,:]) / 1000 # root-relative. milimeter to meter.
        joint_img = data['joint_img']
        joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1)
        joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, joint_cam, data['joint_valid'], do_flip, img_shape, self.joint_set['body']['flip_pairs'], img2bb_trans, rot, self.joint_set['body']['joints_name'], smpl.joints_name)
        
        # smpl coordinates and parameters
        cam_param['t'] /= 1000 # milimeter to meter
        smpl_joint_img, smpl_joint_cam, smpl_joint_trunc, smpl_pose, smpl_shape, smpl_mesh_cam_orig = process_human_model_output(smpl_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smpl')
        smpl_joint_valid = np.ones((smpl.joint_num,1), dtype=np.float32)
        smpl_pose_valid = np.ones((smpl.orig_joint_num*3), dtype=np.float32)
        smpl_shape_valid = float(True)

        """
        # for debug
        #_tmp = joint_img.copy()
        _tmp = smpl_joint_img.copy()
        _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _img = vis_keypoints(_img, _tmp)
        cv2.imwrite('mi_' + str(idx) + '.jpg', _img)
        """

        inputs = {'img': img}
        targets = {'joint_img': joint_img, 'smpl_joint_img': smpl_joint_img, 'joint_cam': joint_cam, 'smpl_joint_cam': smpl_joint_cam, 'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape}
        meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smpl_joint_trunc': smpl_joint_trunc, 'smpl_joint_valid': smpl_joint_valid, 'smpl_pose_valid': smpl_pose_valid, 'smpl_shape_valid': smpl_shape_valid, 'is_3D': float(True)}
        return inputs, targets, meta_info


