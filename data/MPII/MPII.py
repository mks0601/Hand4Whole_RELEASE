import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from utils.human_models import smpl
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
from utils.vis import vis_keypoints, vis_mesh, save_obj

class MPII(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('..', 'data', 'MPII', 'data')
        self.annot_path = osp.join('..', 'data', 'MPII', 'data', 'annotations')

        # mpii skeleton
        self.joint_set = {'body':
                            {'joint_num': 16,
                            'joints_name': ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Thorax', 'Neck', 'Head_top', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist'),
                            'flip_pairs': ( (0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13) )
                            }
                        }
        self.datalist = self.load_data()

    def load_data(self):
        db = COCO(osp.join(self.annot_path, 'train.json'))
        with open(osp.join(self.annot_path, 'MPII_train_SMPL_NeuralAnnot.json')) as f:
            smpl_params = json.load(f)

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            img = db.loadImgs(ann['image_id'])[0]
            imgname = img['file_name']
            img_path = osp.join(self.img_path, imgname)
            
            # bbox
            bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
            if bbox is None: continue
            
            # joint coordinates
            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
            joint_valid = joint_img[:,2:].copy()
            joint_img[:,2] = 0

            smpl_param = smpl_params[str(aid)]

            datalist.append({
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_valid': joint_valid,
                'smpl_param': smpl_param
            })
        return datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # image load and affine transform
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        # mpii gt
        dummy_coord = np.zeros((self.joint_set['body']['joint_num'],3), dtype=np.float32)
        joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(data['joint_img'], dummy_coord, data['joint_valid'], do_flip, img_shape, self.joint_set['body']['flip_pairs'], img2bb_trans, rot, self.joint_set['body']['joints_name'], smpl.joints_name)

        # smpl fitted data
        smpl_param = data['smpl_param']
        smpl_joint_img, smpl_joint_cam, smpl_joint_trunc, smpl_pose, smpl_shape, smpl_mesh_cam_orig = process_human_model_output(smpl_param['smpl_param'], smpl_param['cam_param'], do_flip, img_shape, img2bb_trans, rot, 'smpl')
        smpl_joint_valid = np.ones((smpl.joint_num,1), dtype=np.float32)
        smpl_pose_valid = np.ones((smpl.orig_joint_num*3), dtype=np.float32)
        smpl_shape_valid = float(True)
        
        """
        # for debug
        _tmp = smpl_joint_img.copy() 
        _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _img = vis_keypoints(_img.copy(), _tmp)
        cv2.imwrite('mpii_' + str(idx) + '.jpg', _img)
        """
            
        inputs = {'img': img}
        targets = {'joint_img': joint_img, 'smpl_joint_img': smpl_joint_img, 'joint_cam': joint_cam, 'smpl_joint_cam': smpl_joint_cam, 'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape}
        meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smpl_joint_trunc': smpl_joint_trunc, 'smpl_joint_valid': smpl_joint_valid, 'smpl_pose_valid': smpl_pose_valid, 'smpl_shape_valid': smpl_shape_valid, 'is_3D': float(False)}
        return inputs, targets, meta_info

