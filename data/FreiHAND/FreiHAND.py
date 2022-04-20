import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import mano
from utils.preprocessing import load_img, process_bbox, augmentation, process_human_model_output
from utils.transforms import pixel2cam
from utils.vis import vis_keypoints, vis_mesh, save_obj

class FreiHAND(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'FreiHAND', 'data')
        self.human_bbox_root_dir = osp.join('..', 'data', 'FreiHAND', 'rootnet_output', 'bbox_root_freihand_output.json')
        self.datalist = self.load_data()

    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.data_path, 'freihand_train_coco.json'))
            with open(osp.join(self.data_path, 'freihand_train_data.json')) as f:
                data = json.load(f)
            
        else:
            db = COCO(osp.join(self.data_path, 'freihand_eval_coco.json'))
            with open(osp.join(self.data_path, 'freihand_eval_data.json')) as f:
                data = json.load(f)
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            bbox_root_result = {}
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_root_result[str(annot[i]['image_id'])] = {'bbox': np.array(annot[i]['bbox']), 'root': np.array(annot[i]['root_cam'])}

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.data_path, img['file_name'])
            img_shape = (img['height'], img['width'])
            db_idx = str(img['db_idx'])

            if self.data_split == 'train':
                cam_param, mano_param = data[db_idx]['cam_param'], data[db_idx]['mano_param']
                mano_param['hand_type'] = 'right' # FreiHAND only contains right hand
                bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
                if bbox is None: continue

                datalist.append({
                    'img_path': img_path,
                    'img_shape': img_shape,
                    'bbox': bbox,
                    'cam_param': cam_param,
                    'mano_param': mano_param})
            else:
                cam_param = data[db_idx]['cam_param']
                bbox = bbox_root_result[str(image_id)]['bbox'] # bbox should be aspect ratio preserved-extended. It is done in RootNet.
                root_joint_depth = bbox_root_result[str(image_id)]['root'][2]

                datalist.append({
                    'img_path': img_path,
                    'img_shape': img_shape,
                    'bbox': bbox,
                    'root_depth': root_joint_depth,
                    'cam_param': cam_param})
                
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
        
        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split, enforce_flip=False)
        img = self.transform(img.astype(np.float32))/255.
        
        if self.data_split == 'train':
            # mano coordinates
            cam_param, mano_param = data['cam_param'], data['mano_param']
            mano_joint_img, mano_joint_cam, mano_joint_trunc, mano_pose, mano_shape, mano_mesh_cam_orig = process_human_model_output(mano_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'mano')
            mano_pose_valid = np.ones((mano.orig_joint_num*3), dtype=np.float32)
            mano_joint_valid = np.ones((mano.joint_num,1), dtype=np.float32)

            """
            # for debug
            _tmp = mano_joint_img.copy()
            _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[1] * cfg.input_img_shape[1]
            _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[0] * cfg.input_img_shape[0]
            _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
            _img = vis_keypoints(_img, _tmp)
            cv2.imwrite('frei' + str(idx) + '.jpg', _img)
            """

            inputs = {'img': img}
            targets = {'joint_img': np.zeros_like(mano_joint_img), 'mano_joint_img': mano_joint_img, 'joint_cam': np.zeros_like(mano_joint_cam), 'mano_joint_cam': mano_joint_cam, 'mano_pose': mano_pose, 'mano_shape': mano_shape}
            meta_info = {'joint_valid': np.zeros_like(mano_joint_valid), 'joint_trunc': np.zeros_like(mano_joint_trunc), 'mano_joint_trunc': mano_joint_trunc, 'mano_joint_valid': mano_joint_valid, 'mano_pose_valid': mano_pose_valid, 'mano_shape_valid': float(True), 'is_3D': float(True)}
        else:
            inputs = {'img': img}
            targets = {}
            meta_info = {'bb2img_trans': bb2img_trans}

        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):

        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'joint_out': [], 'mesh_out': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            mesh_out_cam = out['mano_mesh_cam']
            joint_out_cam = np.dot(mano.joint_regressor, mesh_out_cam)

            """
            # positional pose evaluation
            joint_img = out['joint_img']
            joint_img[:,0] = joint_img[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            joint_img[:,1] = joint_img[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,:1])),1)
            joint_img[:,:2] = np.dot(out['bb2img_trans'], joint_img_xy1.transpose(1,0)).transpose(1,0)
            joint_img[:,2] = (joint_img[:,2] / cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size / 2) + annot['root_depth']
            joint_out_cam = pixel2cam(joint_img, annot['cam_param']['focal'], annot['cam_param']['princpt'])
            """

            eval_result['mesh_out'].append(mesh_out_cam.tolist())
            eval_result['joint_out'].append(joint_out_cam.tolist())
 
            vis = False
            if vis:
                filename = annot['img_path'].split('/')[-1][:-4]

                img = load_img(annot['img_path'])[:,:,::-1]
                cv2.imwrite(filename + '.jpg', img)

                save_obj(mesh_out_cam, mano.face['right'], filename + '.obj')

        return eval_result
    
    def print_eval_result(self, eval_result):
        output_save_path = osp.join(cfg.result_dir, 'FreiHAND')
        os.makedirs(output_save_path, exist_ok=True)
        output_save_path = osp.join(output_save_path, 'pred.json')
        with open(output_save_path, 'w') as f:
            json.dump([eval_result['joint_out'], eval_result['mesh_out']], f)
        print('Saved at ' + output_save_path)

