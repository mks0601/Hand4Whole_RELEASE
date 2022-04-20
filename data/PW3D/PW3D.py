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
from utils.preprocessing import load_img, process_bbox, augmentation, process_human_model_output
from utils.transforms import pixel2cam, rigid_align, transform_joint_to_other_db
from utils.vis import vis_keypoints, vis_mesh, save_obj

class PW3D(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'PW3D', 'data')
       
        # H36M joint set
        self.joint_set_h36m = {'body': \
                            {'joint_num': 17,
                            'joints_name': ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'),
                            'eval_joint': (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
                            'smpl_regressor': np.load(osp.join('..', 'data', 'Human36M', 'J_regressor_h36m_smpl.npy'))
                            }
                        }
        self.joint_set_h36m['body']['root_joint_idx'] = self.joint_set_h36m['body']['joints_name'].index('Pelvis')

        self.datalist = self.load_data()

    def load_data(self):
        db = COCO(osp.join(self.data_path, '3DPW_' + self.data_split + '.json'))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            sequence_name = img['sequence']
            img_name = img['file_name']
            img_path = osp.join(self.data_path, 'imageFiles', sequence_name, img_name)
            cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}

            smpl_param = ann['smpl_param']
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue
            data_dict = {'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'], img['width']), 'bbox': bbox, 'smpl_param': smpl_param, 'cam_param': cam_param}
            datalist.append(data_dict)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape = data['img_path'], data['img_shape']
        
        # img
        img = load_img(img_path)
        bbox, smpl_param, cam_param = data['bbox'], data['smpl_param'], data['cam_param']
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.

        # smpl coordinates
        smpl_joint_img, smpl_joint_cam, smpl_joint_trunc, smpl_pose, smpl_shape, smpl_mesh_cam_orig = process_human_model_output(smpl_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smpl')

        inputs = {'img': img}
        targets = {'smpl_mesh_cam': smpl_mesh_cam_orig}
        meta_info = {'bbox': bbox, 'bb2img_trans': bb2img_trans}
        return inputs, targets, meta_info
        
    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': [], 'pa_mpvpe': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
   
            # h36m joint from gt mesh
            mesh_gt_cam = out['smpl_mesh_cam_target']
            pose_coord_gt_h36m = np.dot(self.joint_set_h36m['body']['smpl_regressor'], mesh_gt_cam)
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.joint_set_h36m['body']['root_joint_idx'],None] # root-relative
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.joint_set_h36m['body']['eval_joint'],:]
            mesh_gt_cam -= np.dot(self.joint_set_h36m['body']['smpl_regressor'], mesh_gt_cam)[0,None,:]
            
            # h36m joint from output mesh
            mesh_out_cam = out['smpl_mesh_cam']
            pose_coord_out_h36m = np.dot(self.joint_set_h36m['body']['smpl_regressor'], mesh_out_cam)
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.joint_set_h36m['body']['root_joint_idx'],None] # root-relative
            pose_coord_out_h36m = pose_coord_out_h36m[self.joint_set_h36m['body']['eval_joint'],:]
            pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m)
            eval_result['mpjpe'].append(np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m)**2,1)).mean() * 1000) # meter -> milimeter
            eval_result['pa_mpjpe'].append(np.sqrt(np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m)**2,1)).mean() * 1000) # meter -> milimeter
            mesh_out_cam -= np.dot(self.joint_set_h36m['body']['smpl_regressor'], mesh_out_cam)[0,None,:]
            mesh_out_cam_aligned = rigid_align(mesh_out_cam, mesh_gt_cam)
            eval_result['mpvpe'].append(np.sqrt(np.sum((mesh_out_cam - mesh_gt_cam)**2,1)).mean() * 1000) # meter -> milimeter
            eval_result['pa_mpvpe'].append(np.sqrt(np.sum((mesh_out_cam_aligned - mesh_gt_cam)**2,1)).mean() * 1000) # meter -> milimeter


            vis = False
            if vis:
                file_name = str(cur_sample_idx+n)
                img = (out['img'].transpose(1,2,0)[:,:,::-1] * 255).copy()
                save_obj(mesh_gt_cam, smpl.face, file_name + '_gt.obj')
                save_obj(mesh_out_cam, smpl.face, file_name + '.obj')

                
        return eval_result

    def print_eval_result(self, eval_result):
        print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
        print('PA MPJPE: %.2f mm' % np.mean(eval_result['pa_mpjpe']))
        print('MPVPE: %.2f mm' % np.mean(eval_result['mpvpe']))
        print('PA MPVPE: %.2f mm' % np.mean(eval_result['pa_mpvpe']))




