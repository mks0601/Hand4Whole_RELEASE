import os
import os.path as osp
from glob import glob
import numpy as np
from config import cfg
import copy
import json
import pickle
import cv2
import torch
from pycocotools.coco import COCO
from utils.human_models import smpl_x
from utils.preprocessing import load_img, sanitize_bbox, process_bbox, augmentation, process_db_coord, process_human_model_output, load_ply, load_obj
from utils.transforms import rigid_align
from utils.vis import vis_keypoints, vis_mesh, save_obj, render_mesh

class AGORA(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'AGORA', 'data')
        self.resolution = (2160, 3840) # height, width. one of (720, 1280) and (2160, 3840)
        self.test_set = 'val' # val, test
        self.datalist = self.load_data()
 
    def load_data(self):
        datalist = []
        if self.data_split == 'train' or (self.data_split == 'test' and self.test_set == 'val'): 
            if self.data_split == 'train':
                db = COCO(osp.join(self.data_path, 'AGORA_train_SMPLX.json'))
            else:
                db = COCO(osp.join(self.data_path, 'AGORA_validation_SMPLX.json'))
            
            for aid in db.anns.keys():
                ann = db.anns[aid]
                image_id = ann['image_id']
                person_id = ann['person_id']
                img = db.loadImgs(image_id)[0]
                if not ann['is_valid']:
                    continue
                smplx_param_path = osp.join(self.data_path, ann['smplx_param_path'])
                cam_param_path = osp.join(self.data_path, ann['cam_param_path'])
               
                if self.resolution == (720, 1280):
                    img_shape = self.resolution
                    img_path = osp.join(self.data_path, img['file_name_1280x720'])
                    
                    # convert to current resolution
                    bbox = np.array(ann['bbox']).reshape(2,2)
                    bbox[:,0] = bbox[:,0] / 3840 * 1280
                    bbox[:,1] = bbox[:,1] / 2160 * 720
                    bbox = bbox.reshape(4)
                    bbox = process_bbox(bbox, img_shape[1], img_shape[0])
                    if bbox is None:
                        continue

                    lhand_bbox = np.array(ann['lhand_bbox']).reshape(2,2)
                    lhand_bbox[:,0] = lhand_bbox[:,0] / 3840 * 1280
                    lhand_bbox[:,1] = lhand_bbox[:,1] / 2160 * 720
                    lhand_bbox = lhand_bbox.reshape(4)
                    lhand_bbox = sanitize_bbox(lhand_bbox, img_shape[1], img_shape[0])
                    if lhand_bbox is not None:
                        lhand_bbox[2:] += lhand_bbox[:2] # xywh -> xyxy

                    rhand_bbox = np.array(ann['rhand_bbox']).reshape(2,2)
                    rhand_bbox[:,0] = rhand_bbox[:,0] / 3840 * 1280
                    rhand_bbox[:,1] = rhand_bbox[:,1] / 2160 * 720
                    rhand_bbox = rhand_bbox.reshape(4)
                    rhand_bbox = sanitize_bbox(rhand_bbox, img_shape[1], img_shape[0])
                    if rhand_bbox is not None:
                        rhand_bbox[2:] += rhand_bbox[:2] # xywh -> xyxy

                    face_bbox = np.array(ann['face_bbox']).reshape(2,2)
                    face_bbox[:,0] = face_bbox[:,0] / 3840 * 1280
                    face_bbox[:,1] = face_bbox[:,1] / 2160 * 720
                    face_bbox = face_bbox.reshape(4)
                    face_bbox = sanitize_bbox(face_bbox, img_shape[1], img_shape[0])
                    if face_bbox is not None:
                        face_bbox[2:] += face_bbox[:2] # xywh -> xyxy

                    data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'lhand_bbox': lhand_bbox, 'rhand_bbox': rhand_bbox, 'face_bbox': face_bbox, 'smplx_param_path': smplx_param_path, 'cam_param_path': cam_param_path, 'ann_id': str(aid)}
                    datalist.append(data_dict)

                elif self.resolution == (2160, 3840): # use cropped and resized images. loading 4K images in pytorch dataloader takes too much time...
                    img_path = osp.join(self.data_path, 'images_3840x2160', img['file_name_3840x2160'].split('/')[-2] + '_crop', img['file_name_3840x2160'].split('/')[-1][:-4] + '_person_id_' + str(person_id) + '.png')
                    json_path = osp.join(self.data_path, 'images_3840x2160', img['file_name_3840x2160'].split('/')[-2] + '_crop', img['file_name_3840x2160'].split('/')[-1][:-4] + '_person_id_' + str(person_id) + '.json')
                    if not osp.isfile(json_path):
                        continue
                    with open(json_path) as f:
                        crop_resize_info = json.load(f)
                        img2bb_trans_from_orig = np.array(crop_resize_info['affine_mat'], dtype=np.float32)
                        resized_height, resized_width = crop_resize_info['resized_height'], crop_resize_info['resized_width']
                    img_shape = (resized_height, resized_width)
                    bbox = np.array([0, 0, resized_width, resized_height], dtype=np.float32)
                    
                    # transform from original image to crop_and_resize image
                    lhand_bbox = np.array(ann['lhand_bbox']).reshape(2,2)
                    lhand_bbox[1] += lhand_bbox[0] # xywh -> xyxy
                    lhand_bbox = np.dot(img2bb_trans_from_orig, np.concatenate((lhand_bbox, np.ones_like(lhand_bbox[:,:1])),1).transpose(1,0)).transpose(1,0) 
                    lhand_bbox[1] -= lhand_bbox[0] # xyxy -> xywh
                    lhand_bbox = lhand_bbox.reshape(4)
                    lhand_bbox = sanitize_bbox(lhand_bbox, self.resolution[1], self.resolution[0])
                    if lhand_bbox is not None:
                        lhand_bbox[2:] += lhand_bbox[:2] # xywh -> xyxy

                    # transform from original image to crop_and_resize image
                    rhand_bbox = np.array(ann['rhand_bbox']).reshape(2,2)
                    rhand_bbox[1] += rhand_bbox[0] # xywh -> xyxy
                    rhand_bbox = np.dot(img2bb_trans_from_orig, np.concatenate((rhand_bbox, np.ones_like(rhand_bbox[:,:1])),1).transpose(1,0)).transpose(1,0) 
                    rhand_bbox[1] -= rhand_bbox[0] # xyxy -> xywh
                    rhand_bbox = rhand_bbox.reshape(4)
                    rhand_bbox = sanitize_bbox(rhand_bbox, self.resolution[1], self.resolution[0])
                    if rhand_bbox is not None:
                        rhand_bbox[2:] += rhand_bbox[:2] # xywh -> xyxy

                    # transform from original image to crop_and_resize image
                    face_bbox = np.array(ann['face_bbox']).reshape(2,2)
                    face_bbox[1] += face_bbox[0] # xywh -> xyxy
                    face_bbox = np.dot(img2bb_trans_from_orig, np.concatenate((face_bbox, np.ones_like(face_bbox[:,:1])),1).transpose(1,0)).transpose(1,0) 
                    face_bbox[1] -= face_bbox[0] # xyxy -> xywh
                    face_bbox = face_bbox.reshape(4)
                    face_bbox = sanitize_bbox(face_bbox, self.resolution[1], self.resolution[0])
                    if face_bbox is not None:
                        face_bbox[2:] += face_bbox[:2] # xywh -> xyxy

                    data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'lhand_bbox': lhand_bbox, 'rhand_bbox': rhand_bbox, 'face_bbox': face_bbox, 'img2bb_trans_from_orig': img2bb_trans_from_orig, 'smplx_param_path': smplx_param_path, 'cam_param_path': cam_param_path, 'ann_id': str(aid)}
                    datalist.append(data_dict)

        elif self.data_split == 'test' and self.test_set == 'test':
            with open(osp.join(self.data_path, 'AGORA_test_bbox.json')) as f:
                bboxs = json.load(f)

            for filename in bboxs.keys():
                if self.resolution == (720, 1280):
                    img_path = osp.join(self.data_path, 'test', filename)
                    img_shape = self.resolution
                    person_num = len(bboxs[filename])
                    for person_id in range(person_num):
                        # change bbox from (2160,3840) to target resoution
                        bbox = np.array(bboxs[filename][person_id]['bbox']).reshape(2,2)
                        bbox[:,0] = bbox[:,0] / 3840 * 1280
                        bbox[:,1] = bbox[:,1] / 2160 * 720
                        bbox = bbox.reshape(4)
                        bbox = process_bbox(bbox, img_shape[1], img_shape[0])
                        if bbox is None:
                            continue
                        datalist.append({'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'person_id': person_id})

                elif self.resolution == (2160, 3840): # use cropped and resized images. loading 4K images in pytorch dataloader takes too much time...
                    person_num = len(bboxs[filename])
                    for person_id in range(person_num):
                        img_path = osp.join(self.data_path, 'images_3840x2160', 'test_crop', filename[:-4] + '_person_id_' + str(person_id) + '.png')
                        json_path = osp.join(self.data_path, 'images_3840x2160', 'test_crop', filename[:-4] + '_person_id_' + str(person_id) + '.json')
                        if not osp.isfile(json_path):
                            continue
                        with open(json_path) as f:
                            crop_resize_info = json.load(f)
                            img2bb_trans_from_orig = np.array(crop_resize_info['img2bb_trans'], dtype=np.float32)
                            resized_height, resized_width = crop_resize_info['resized_height'], crop_resize_info['resized_width']
                        img_shape = (resized_height, resized_width)
                        bbox = np.array([0, 0, resized_width, resized_height], dtype=np.float32)
                        datalist.append({'img_path': img_path, 'img_shape': img_shape, 'img2bb_trans_from_orig': img2bb_trans_from_orig, 'bbox': bbox, 'person_id': person_id})

        return datalist

    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0,0,1,1], dtype=np.float32).reshape(2,2) # dummy value
            bbox_valid = float(False) # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2,2) 

            # flip augmentation
            if do_flip:
                bbox[:,0] = img_shape[1] - bbox[:,0] - 1
                bbox[0,0], bbox[1,0] = bbox[1,0].copy(), bbox[0,0].copy() # xmin <-> xmax swap
            
            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4,2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:,:1])),1) 
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            bbox[:,0] = bbox[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            bbox[:,1] = bbox[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            # make box a rectangle without rotation
            xmin = np.min(bbox[:,0]); xmax = np.max(bbox[:,0]);
            ymin = np.min(bbox[:,1]); ymax = np.max(bbox[:,1]);
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            
            bbox_valid = float(True)
            bbox = bbox.reshape(2,2)

        return bbox, bbox_valid

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

        if self.data_split == 'train':
            # hand and face bbox transform
            lhand_bbox, rhand_bbox, face_bbox = data['lhand_bbox'], data['rhand_bbox'], data['face_bbox']
            lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(lhand_bbox, do_flip, img_shape, img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(rhand_bbox, do_flip, img_shape, img2bb_trans)
            face_bbox, face_bbox_valid = self.process_hand_face_bbox(face_bbox, do_flip, img_shape, img2bb_trans)
            if do_flip:
                lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1])/2.; rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1])/2.; face_bbox_center = (face_bbox[0] + face_bbox[1])/2.
            lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]; rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]; face_bbox_size = face_bbox[1] - face_bbox[0];
            
            """
            # for debug
            _img = img.numpy().transpose(1,2,0)[:,:,::-1].copy() * 255
            if lhand_bbox_valid:
                _tmp = lhand_bbox.copy().reshape(2,2)
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                cv2.rectangle(_img, (int(_tmp[0,0]), int(_tmp[0,1])), (int(_tmp[1,0]), int(_tmp[1,1])), (255,0,0), 3)
                cv2.imwrite('agora_' + str(idx) + '_lhand.jpg', _img)
            if rhand_bbox_valid:
                _tmp = rhand_bbox.copy().reshape(2,2)
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                cv2.rectangle(_img, (int(_tmp[0,0]), int(_tmp[0,1])), (int(_tmp[1,0]), int(_tmp[1,1])), (255,0,0), 3)
                cv2.imwrite('agora_' + str(idx) + '_rhand.jpg', _img)
            if face_bbox_valid:
                _tmp = face_bbox.copy().reshape(2,2)
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                cv2.rectangle(_img, (int(_tmp[0,0]), int(_tmp[0,1])), (int(_tmp[1,0]), int(_tmp[1,1])), (255,0,0), 3)
                cv2.imwrite('agora_' + str(idx) + '_face.jpg', _img)
            #cv2.imwrite('agora_' + str(idx) + '.jpg', _img)
            """
            
            # smplx parameters
            with open(data['smplx_param_path']) as f:
                smplx_param = json.load(f)
            root_pose = np.array(smplx_param['global_orient'], dtype=np.float32).reshape(-1) 
            body_pose = np.array(smplx_param['body_pose'], dtype=np.float32).reshape(-1)
            shape = np.array(smplx_param['betas'], dtype=np.float32).reshape(-1)[:10]
            lhand_pose = np.array(smplx_param['left_hand_pose'], dtype=np.float32).reshape(-1)
            rhand_pose = np.array(smplx_param['right_hand_pose'], dtype=np.float32).reshape(-1)
            jaw_pose = np.array(smplx_param['jaw_pose'], dtype=np.float32).reshape(-1)
            expr = np.array(smplx_param['expression'], dtype=np.float32).reshape(-1)
            trans = np.array(smplx_param['transl'], dtype=np.float32).reshape(-1) 
            with open(data['cam_param_path']) as f:
                cam_param = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}            
            if self.resolution == (2160, 3840): # apply crop and resize
                cam_param['focal'][0] = cam_param['focal'][0] * data['img2bb_trans_from_orig'][0][0]
                cam_param['focal'][1] = cam_param['focal'][1] * data['img2bb_trans_from_orig'][1][1]
                cam_param['princpt'][0] = cam_param['princpt'][0] * data['img2bb_trans_from_orig'][0][0] + data['img2bb_trans_from_orig'][0][2]
                cam_param['princpt'][1] = cam_param['princpt'][1] * data['img2bb_trans_from_orig'][1][1] + data['img2bb_trans_from_orig'][1][2]
            else: # scale camera parameters
                cam_param['focal'][0] = cam_param['focal'][0] / 3840 * self.resolution[1]
                cam_param['focal'][1] = cam_param['focal'][1] / 2160 * self.resolution[0]
                cam_param['princpt'][0] = cam_param['princpt'][0] / 3840 * self.resolution[1]
                cam_param['princpt'][1] = cam_param['princpt'][1] / 2160 * self.resolution[0]
            smplx_param = {'root_pose': root_pose, 'body_pose': body_pose, 'shape': shape,
                    'lhand_pose': lhand_pose, 'lhand_valid': True, 
                    'rhand_pose': rhand_pose, 'rhand_valid': True, 
                    'jaw_pose': jaw_pose, 'expr': expr, 'face_valid': True,
                    'trans': trans}
            smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')
            smplx_pose_valid = np.tile(smplx_pose_valid[:,None], (1,3)).reshape(-1)
            smplx_shape_valid = True

            inputs = {'img': img}
            targets = {'joint_img': smplx_joint_img, 'joint_cam': smplx_joint_cam, 'smplx_joint_img': smplx_joint_img, 'smplx_joint_cam': smplx_joint_cam, 'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape, 'smplx_expr': smplx_expr, 'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size, 'face_bbox_center': face_bbox_center, 'face_bbox_size': face_bbox_size}
            meta_info = {'joint_valid': smplx_joint_valid, 'joint_trunc': smplx_joint_trunc, 'smplx_joint_valid': smplx_joint_valid, 'smplx_joint_trunc': smplx_joint_trunc, 'smplx_pose_valid': smplx_pose_valid, 'smplx_shape_valid': float(smplx_shape_valid), 'smplx_expr_valid': float(smplx_expr_valid), 'is_3D': float(True), 'lhand_bbox_valid': lhand_bbox_valid, 'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid}
            return inputs, targets, meta_info
        else:
            # transform from original image to crop_and_resize image
            if self.resolution == (2160, 3840):
                mat1 = np.concatenate((data['img2bb_trans_from_orig'], np.array([0,0,1], dtype=np.float32).reshape(1,3)))
                mat2 = np.concatenate((img2bb_trans, np.array([0,0,1], dtype=np.float32).reshape(1,3)))
                img2bb_trans = np.dot(mat2, mat1)[:2,:3]
                bb2img_trans = np.dot(np.linalg.inv(mat1), np.linalg.inv(mat2))[:2,:3]    
                
            if self.test_set == 'val':
                # smplx parameters
                with open(data['smplx_param_path']) as f:
                    smplx_param = json.load(f)
                root_pose = np.array(smplx_param['global_orient'], dtype=np.float32).reshape(-1) 
                body_pose = np.array(smplx_param['body_pose'], dtype=np.float32).reshape(-1)
                shape = np.array(smplx_param['betas'], dtype=np.float32).reshape(-1)[:10]
                lhand_pose = np.array(smplx_param['left_hand_pose'], dtype=np.float32).reshape(-1)
                rhand_pose = np.array(smplx_param['right_hand_pose'], dtype=np.float32).reshape(-1)
                jaw_pose = np.array(smplx_param['jaw_pose'], dtype=np.float32).reshape(-1)
                expr = np.array(smplx_param['expression'], dtype=np.float32).reshape(-1)
                trans = np.array(smplx_param['transl'], dtype=np.float32).reshape(-1) 
                with open(data['cam_param_path']) as f:
                    cam_param = json.load(f)
                smplx_param = {'root_pose': root_pose, 'body_pose': body_pose, 'shape': shape,
                        'lhand_pose': lhand_pose, 'lhand_valid': True, 
                        'rhand_pose': rhand_pose, 'rhand_valid': True, 
                        'jaw_pose': jaw_pose, 'expr': expr, 'face_valid': True,
                        'trans': trans}
                _, _, _, _, _, _, _, _, _, smplx_mesh_cam_orig = process_human_model_output(smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')

                inputs = {'img': img}
                targets = {'smplx_mesh_cam': smplx_mesh_cam_orig}
                meta_info = {'bb2img_trans': bb2img_trans}
            else:
                inputs = {'img': img}
                targets = {'smplx_mesh_cam': np.zeros((smpl_x.vertex_num, 3), dtype=np.float32)} # dummy vertex
                meta_info = {'bb2img_trans': bb2img_trans}

            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'pa_mpvpe_all': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 'mpvpe_all': [], 'mpvpe_hand': [], 'mpvpe_face': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            mesh_gt = out['smplx_mesh_cam_target']
            mesh_out = out['smplx_mesh_cam']
           
            # MPVPE from all vertices
            mesh_out_align = mesh_out - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['pelvis'],None,:] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['pelvis'],None,:]
            eval_result['mpvpe_all'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt)**2,1)).mean() * 1000)
            mesh_out_align = rigid_align(mesh_out, mesh_gt)
            eval_result['pa_mpvpe_all'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt)**2,1)).mean() * 1000)
 
            # MPVPE from hand vertices
            mesh_gt_lhand = mesh_gt[smpl_x.hand_vertex_idx['left_hand'],:]
            mesh_out_lhand = mesh_out[smpl_x.hand_vertex_idx['left_hand'],:]
            mesh_gt_rhand = mesh_gt[smpl_x.hand_vertex_idx['right_hand'],:]
            mesh_out_rhand = mesh_out[smpl_x.hand_vertex_idx['right_hand'],:]
            mesh_out_lhand_align = mesh_out_lhand - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['lwrist'],None,:] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['lwrist'],None,:]
            mesh_out_rhand_align = mesh_out_rhand - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['rwrist'],None,:] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['rwrist'],None,:]
            eval_result['mpvpe_hand'].append((np.sqrt(np.sum((mesh_out_lhand_align - mesh_gt_lhand)**2,1)).mean() * 1000 + np.sqrt(np.sum((mesh_out_rhand_align - mesh_gt_rhand)**2,1)).mean() * 1000)/2.)
            mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
            mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)
            eval_result['pa_mpvpe_hand'].append((np.sqrt(np.sum((mesh_out_lhand_align - mesh_gt_lhand)**2,1)).mean() * 1000 + np.sqrt(np.sum((mesh_out_rhand_align - mesh_gt_rhand)**2,1)).mean() * 1000)/2.)

            # MPVPE from face vertices
            mesh_gt_face = mesh_gt[smpl_x.face_vertex_idx,:]
            mesh_out_face = mesh_out[smpl_x.face_vertex_idx,:]
            mesh_out_face_align = mesh_out_face - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['neck'],None,:] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['neck'],None,:]
            eval_result['mpvpe_face'].append(np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face)**2,1)).mean() * 1000)
            mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
            eval_result['pa_mpvpe_face'].append(np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face)**2,1)).mean() * 1000)
           
            vis = False
            if vis:
                #img = (out['img'].transpose(1,2,0)[:,:,::-1] * 255).copy()
                #joint_img = out['joint_img'].copy()
                #joint_img[:,0] = joint_img[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                #joint_img[:,1] = joint_img[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                #for j in range(len(joint_img)):
                #    cv2.circle(img, (int(joint_img[j][0]), int(joint_img[j][1])), 3, (0,0,255), -1)
                #cv2.imwrite(str(cur_sample_idx + n) + '.jpg', img)

                img_path = annot['img_path']
                img_id = img_path.split('/')[-1][:-4]
                ann_id = annot['ann_id']
                img = load_img(img_path)[:,:,::-1]
                bbox = annot['bbox']
                focal = list(cfg.focal)
                princpt = list(cfg.princpt)
                focal[0] = focal[0] / cfg.input_body_shape[1] * bbox[2]
                focal[1] = focal[1] / cfg.input_body_shape[0] * bbox[3]
                princpt[0] = princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0]
                princpt[1] = princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]
                img = render_mesh(img, out['smplx_mesh_cam'], smpl_x.face, {'focal': focal, 'princpt': princpt})
                #img = cv2.resize(img, (512,512))
                cv2.imwrite(img_id + '_' + str(ann_id) + '.jpg', img)


                vis_mesh_out = out['smplx_mesh_cam']
                vis_mesh_out = vis_mesh_out - np.dot(smpl_x.layer['neutral'].J_regressor, vis_mesh_out)[smpl_x.J_regressor_idx['pelvis'],None,:] 
                #vis_mesh_gt = out['smplx_mesh_cam_target']
                #vis_mesh_gt = vis_mesh_gt - np.dot(smpl_x.layer['neutral'].J_regressor, vis_mesh_gt)[smpl_x.J_regressor_idx['pelvis'],None,:] 
                save_obj(vis_mesh_out, smpl_x.face, img_id + '_' + str(ann_id) + '.obj')
                #save_obj(vis_mesh_gt, smpl_x.face, str(cur_sample_idx + n) + '_gt.obj')

            
            # save results for the official evaluation codes/server
            save_name = annot['img_path'].split('/')[-1][:-4]
            if self.data_split == 'test' and self.test_set == 'test':
                if self.resolution == (2160,3840):
                    save_name = save_name.split('_person_id')[0]
            elif self.data_split == 'test' and self.test_set == 'val':
                if self.resolution == (2160,3840):
                    save_name = save_name.split('_person_id')[0]
                else:
                    save_name = save_name.split('_1280x720')[0]
            if 'person_id' in annot:
                person_id = annot['person_id']
            else:
                exist_result_path = glob(osp.join(cfg.result_dir, 'AGORA', save_name + '*'))
                if len(exist_result_path) == 0:
                    person_id = 0
                else:
                    last_person_id = max([int(name.split('personId_')[1].split('.pkl')[0]) for name in exist_result_path])
                    person_id = last_person_id + 1
            save_name += '_personId_' + str(person_id) + '.pkl'

            joint_proj = out['smplx_joint_proj']
            joint_proj[:,0] = joint_proj[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            joint_proj[:,1] = joint_proj[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            joint_proj = np.concatenate((joint_proj, np.ones_like(joint_proj[:,:1])),1)
            joint_proj = np.dot(out['bb2img_trans'], joint_proj.transpose(1,0)).transpose(1,0)
            joint_proj[:,0] = joint_proj[:,0] / self.resolution[1] * 3840 # restore to original resolution
            joint_proj[:,1] = joint_proj[:,1] / self.resolution[0] * 2160 # restore to original resolution
            save_dict = {'params': 
                                {'transl': out['cam_trans'].reshape(1,-1),
                                'global_orient': out['smplx_root_pose'].reshape(1,-1),
                                'body_pose': out['smplx_body_pose'].reshape(1,-1),
                                'left_hand_pose': out['smplx_lhand_pose'].reshape(1,-1),
                                'right_hand_pose': out['smplx_rhand_pose'].reshape(1,-1),
                                'reye_pose': np.zeros((1,3)),
                                'leye_pose': np.zeros((1,3)),
                                'jaw_pose': out['smplx_jaw_pose'].reshape(1,-1),
                                'expression': out['smplx_expr'].reshape(1,-1),
                                'betas': out['smplx_shape'].reshape(1,-1)},
                        'joints': joint_proj.reshape(1,-1,2)
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

        print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
        print('PA MPVPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_hand']))
        print('PA MPVPE (Face): %.2f mm' % np.mean(eval_result['pa_mpvpe_face']))

        print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        print('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        print('MPVPE (Face): %.2f mm' % np.mean(eval_result['mpvpe_face']))


