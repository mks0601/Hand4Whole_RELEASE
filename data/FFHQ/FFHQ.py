import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import flame
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
from utils.vis import vis_keypoints, vis_mesh, save_obj

class FFHQ(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'FFHQ', 'data')
        self.datalist = self.load_data()
        
    def load_data(self):
        db = COCO(osp.join(self.data_path, 'FFHQ.json'))
        with open(osp.join(self.data_path, 'flame_param.json')) as f:
            flame_params = json.load(f)
            
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.data_path, 'images', img['file_name'])
            img_shape = (img['height'], img['width'])
            
            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,2)
            joint_img = np.concatenate((joint_img, np.zeros_like(joint_img[:,:1])),1)
            joint_valid = np.array(ann['keypoints_valid'], dtype=np.float32).reshape(-1,1)
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue
           
            flame_param = flame_params[str(aid)]

            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_valid': joint_valid,
                'flame_param': flame_param,
                'ann_id': aid})

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
        
        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        # ffhq gt
        dummy_coord = np.zeros((flame.joint_num,3), dtype=np.float32)
        joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(data['joint_img'], dummy_coord, data['joint_valid'], do_flip, img_shape, flame.flip_pairs, img2bb_trans, rot, flame.joints_name, flame.joints_name)

        # flame fitted data
        flame_param = data['flame_param']
        flame_joint_img, flame_joint_cam, flame_joint_trunc, flame_root_pose, flame_jaw_pose, flame_shape, flame_expr, flame_joint_cam_orig, flame_mesh_cam_orig = process_human_model_output(flame_param['flame_param'], flame_param['cam_param'], do_flip, img_shape, img2bb_trans, rot, 'flame')
        flame_joint_valid = np.ones((flame.joint_num,1), dtype=np.float32)
        flame_root_pose_valid = float(True)
        flame_jaw_pose_valid = float(True)
        flame_shape_valid = float(True)
        flame_expr_valid = float(True)

        """ 
        # for debug
        #_tmp = flame_joint_img.copy()
        _tmp = joint_img.copy()
        _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[1] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[0] * cfg.input_img_shape[0]
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _img = vis_keypoints(_img, _tmp)
        cv2.imwrite('ffhq_' + str(idx) + '.jpg', _img)
        """

        inputs = {'img': img}
        targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'flame_joint_cam': flame_joint_cam, 'flame_root_pose': flame_root_pose, 'flame_jaw_pose': flame_jaw_pose, 'flame_shape': flame_shape, 'flame_expr': flame_expr}
        meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'flame_joint_valid': flame_joint_valid, 'flame_root_pose_valid': flame_root_pose_valid, 'flame_jaw_pose_valid': flame_jaw_pose_valid, 'flame_shape_valid': flame_shape_valid, 'flame_expr_valid': flame_expr_valid, 'is_3D': float(False)}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            vis = False
            if vis:
                ann_id = annot['ann_id']
                #filename = annot['img_path'].split('/')[-1][:-4]
                #img = load_img(annot['img_path'])[:,:,::-1]
                #cv2.imwrite(filename + '.jpg', img)
                #save_obj(out['flame_mesh_cam'], flame.face, filename + '.obj')

                bbox = annot['bbox']
                flame_root_pose = out['flame_root_pose']; flame_jaw_pose = out['flame_jaw_pose']; flame_shape = out['flame_shape']; flame_expr = out['flame_expr']; flame_trans = out['cam_trans']
                focal_x = cfg.focal[0] / cfg.input_img_shape[1] * bbox[2]
                focal_y = cfg.focal[1] / cfg.input_img_shape[0] * bbox[3]
                princpt_x = cfg.princpt[0] / cfg.input_img_shape[1] * bbox[2] + bbox[0]
                princpt_y = cfg.princpt[1] / cfg.input_img_shape[0] * bbox[3] + bbox[1]
                save_dict = {'flame_param': {'root_pose': flame_root_pose.reshape(-1).tolist(), 'jaw_pose': flame_jaw_pose.reshape(-1).tolist(), 'shape': flame_shape.reshape(-1).tolist(), 'expr': flame_expr.reshape(-1).tolist(), 'trans': flame_trans.reshape(-1).tolist()},\
                            'cam_param': {'focal': (focal_x,focal_y), 'princpt': (princpt_x,princpt_y)}
                            }
                with open(osp.join(cfg.result_dir, 'flame_param_' + str(ann_id) + '.json'), 'w') as f:
                    json.dump(save_dict, f)


                
        return eval_result

    def print_eval_result(self, eval_result):
        return
