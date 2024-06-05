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
        assert cfg.parts == 'body' # only support body
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'AGORA', 'data')
        self.resolution = (2160, 3840) # height, width. one of (720, 1280) and (2160, 3840)
        self.test_set = 'test' # val, test
        
        # AGORA joint set
        self.joint_set = {
                            'joint_num': 45,
                            'joints_name': ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4', 'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4'),
                            'flip_pairs': ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23), (25,26), (27,28), (29,32), (30,33), (31,34), (35,40), (36,41), (37,42), (38,43), (39,44) )           
                            }
        self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index('Pelvis')
        self.datalist = self.load_data()

    def load_data(self):
        datalist = []

        if self.data_split == 'train' or (self.data_split == 'test' and self.test_set == 'val'):
            if self.data_split == 'train':
                db = COCO(osp.join(self.data_path, 'AGORA_train_SMPL.json'))
            else:
                db = COCO(osp.join(self.data_path, 'AGORA_validation_SMPL.json'))
            
            for aid in db.anns.keys():
                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                if not ann['is_valid']:
                    continue

                if self.resolution == (720, 1280):
                    img_shape = (img['height'], img['width'])
                    img_path = osp.join(self.data_path, img['file_name_1280x720'])
                    smpl_param_path = osp.join(self.data_path, ann['smpl_param_path'])
                    cam_param_path = osp.join(self.data_path, ann['cam_param_path'])

                    # convert to current resolution
                    bbox = np.array(ann['bbox']).reshape(4)
                    bbox[:,0] = bbox[:,0] / 3840 * 1280
                    bbox[:,1] = bbox[:,1] / 2160 * 720
                    bbox = bbox.reshape(4)
                    bbox = process_bbox(bbox, img_shape[1], img_shape[0])
                    if bbox is None:
                        continue

                    data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'smpl_param_path': smpl_param_path, 'cam_param_path': cam_param_path}
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

                    smpl_param_path = osp.join(self.data_path, ann['smpl_param_path'])
                    cam_param_path = osp.join(self.data_path, ann['cam_param_path'])
                    bbox = np.array([0, 0, resized_width, resized_height], dtype=np.float32)
                    data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'img2bb_trans_from_orig': img2bb_trans_from_orig, 'smpl_param_path': smpl_param_path, 'cam_param_path': cam_param_path}
                    datalist.append(data_dict)

        elif self.data_split == 'test' and self.test_set == 'test': 
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
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split) 
        img = self.transform(img.astype(np.float32))/255.

        # transform from original image to crop_and_resize image
        if self.resolution == (2160, 3840):
            mat1 = np.concatenate((data['img2bb_trans_from_orig'], np.array([0,0,1], dtype=np.float32)))
            mat2 = np.concatenate((img2bb_trans, np.array([0,0,1], dtype=np.float32)))
            img2bb_trans = np.dot(mat2, mat1)[:2,:3]
            bb2img_trans = np.dot(np.linalg.inv(mat1), np.linalg.inv(mat2))[:2,:3]
                
        # train mode
        if self.data_split == 'train':
            # load smpl and camera parameters
            with open(data['smpl_param_path']) as f:
                smpl_param = json.load(f)
            with open(data['cam_param_path']) as f:
                cam_param = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()} # 'focal', 'princpt'
            # scale camera parameters
            if self.resolution != (2160, 3840):
                cam_param['focal'][0] = cam_param['focal'][0] / 3840 * self.resolution[1]
                cam_param['focal'][1] = cam_param['focal'][1] / 2160 * self.resolution[0]
                cam_param['princpt'][0] = cam_param['princpt'][0] / 3840 * self.resolution[1]
                cam_param['princpt'][1] = cam_param['princpt'][1] / 2160 * self.resolution[0]

            # smpl parameters
            root_pose = np.array(smpl_param['global_orient'], dtype=np.float32).reshape(-1) 
            body_pose = np.array(smpl_param['body_pose'], dtype=np.float32).reshape(-1)
            smpl_pose = np.concatenate((root_pose, body_pose))
            shape = np.array(smpl_param['betas'], dtype=np.float32).reshape(-1)
            trans = np.array(smpl_param['transl'], dtype=np.float32).reshape(-1) 
            smpl_param = {'pose': smpl_pose, 'shape': shape, 'trans': trans}
            smpl_joint_img, smpl_joint_cam, smpl_joint_trunc, smpl_pose, smpl_shape, smpl_mesh_cam_orig = process_human_model_output(smpl_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smpl')
            smpl_joint_valid = np.ones((smpl.joint_num,1), dtype=np.float32)
            smpl_pose_valid = np.ones((smpl.orig_joint_num*3), dtype=np.float32)
            
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
            targets = {'joint_img': smpl_joint_img, 'joint_cam': smpl_joint_cam, 'smpl_joint_img': smpl_joint_img, 'smpl_joint_cam': smpl_joint_cam, 'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape}
            meta_info = {'joint_valid': smpl_joint_valid, 'joint_trunc': smpl_joint_trunc, 'smpl_joint_valid': smpl_joint_valid, 'smpl_joint_trunc': smpl_joint_trunc, 'smpl_pose_valid': smpl_pose_valid, 'smpl_shape_valid': float(True), 'is_3D': float(True)}
            return inputs, targets, meta_info

        # test mode
        else:
            if self.test_set == 'val':
                # load smpl and camera parameters
                with open(data['smpl_param_path']) as f:
                    smpl_param = json.load(f)
                with open(data['cam_param_path']) as f:
                    cam_param = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()} # 'focal', 'princpt'
                # scale camera parameters
                if self.resolution != (2160, 3840):
                    cam_param['focal'][0] = cam_param['focal'][0] / 3840 * self.resolution[1]
                    cam_param['focal'][1] = cam_param['focal'][1] / 2160 * self.resolution[0]
                    cam_param['princpt'][0] = cam_param['princpt'][0] / 3840 * self.resolution[1]
                    cam_param['princpt'][1] = cam_param['princpt'][1] / 2160 * self.resolution[0]
    
                # smpl parameters
                root_pose = np.array(smpl_param['global_orient'], dtype=np.float32).reshape(-1) 
                body_pose = np.array(smpl_param['body_pose'], dtype=np.float32).reshape(-1)
                smpl_pose = np.concatenate((root_pose, body_pose))
                shape = np.array(smpl_param['betas'], dtype=np.float32).reshape(-1)
                trans = np.array(smpl_param['transl'], dtype=np.float32).reshape(-1) 
                smpl_param = {'pose': smpl_pose, 'shape': shape, 'trans': trans}
                smpl_joint_img, smpl_joint_cam, smpl_joint_trunc, smpl_pose, smpl_shape, smpl_mesh_cam_orig = process_human_model_output(smpl_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smpl')

                inputs = {'img': img}
                targets = {'smpl_mesh_cam': smpl_mesh_cam_orig}
                meta_info = {'bb2img_trans': bb2img_trans}
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
            mesh_gt = out['smpl_mesh_cam_target']
            mesh_out = out['smpl_mesh_cam']
           
            mesh_out_align = mesh_out - np.dot(smpl.joint_regressor, mesh_out)[smpl.root_joint_idx,None,:] + np.dot(smpl.joint_regressor, mesh_gt)[smpl.root_joint_idx,None,:]
            eval_result['mpvpe'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt)**2,1)).mean() * 1000)
            mesh_out_align = rigid_align(mesh_out, mesh_gt)
            eval_result['pa_mpvpe'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt)**2,1)).mean() * 1000)
            
            vis = False
            if vis:
                img = (out['img'].transpose(1,2,0)[:,:,::-1] * 255).copy()
                cv2.imwrite(str(cur_sample_idx + n) + '.jpg', img)
                
                vis_mesh_out = out['smpl_mesh_cam']
                vis_mesh_out = vis_mesh_out - np.dot(smpl.joint_regressor, vis_mesh_out)[smpl.root_joint_idx,None,:]
                vis_mesh_gt = out['smpl_mesh_cam_target']
                vis_mesh_gt = vis_mesh_gt - np.dot(smpl.joint_regressor, vis_mesh_gt)[smpl.root_joint_idx,None,:] 
                save_obj(vis_mesh_out, smpl.face, str(cur_sample_idx + n) + '.obj')
                save_obj(vis_mesh_gt, smpl.face, str(cur_sample_idx + n) + '_gt.obj')

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


