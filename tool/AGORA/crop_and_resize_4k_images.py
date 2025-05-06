import json
import torch
import numpy as np
import os.path as osp
from glob import glob
from tqdm import tqdm
import cv2
import os
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, dest='dataset_path')
    parser.add_argument('--out_height', type=str, dest='out_height')
    parser.add_argument('--out_width', type=str, dest='out_width')
    args = parser.parse_args()
    
    if not args.dataset_path:
        assert 0, "Please set dataset_path"

    if not args.out_height or not args.out_width:
        assert 0, "Please set output (height and width. For example, --out_height 512 --out_width 384"

    return args

def set_aspect_ratio(bbox, aspect_ratio):
   # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox

def get_patch_img(cvimg, bbox, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
   
    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans = get_affine_trans_mat(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0])
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = get_affine_trans_mat(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], inv=True)
    return img_patch, trans, inv_trans

def get_affine_trans_mat(c_x, c_y, src_width, src_height, dst_width, dst_height, inv=False):
    src_w = src_width
    src_h = src_height
    src_center = np.array([c_x, c_y], dtype=np.float32)

    src_downdir = np.array([0, src_h * 0.5], dtype=np.float32)
    src_rightdir = np.array([src_w * 0.5, 0], dtype=np.float32)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

class AGORA(torch.utils.data.Dataset):
    def __init__(self, dataset_path, out_height, out_width):
        self.root_path = dataset_path
        self.img_shape = (2160, 3840) # height, width
        self.out_shape = (out_height, out_width)
        
        self.datalist = []
        for split in ('train', 'validation', 'test'):
           
            if split in ('train', 'validation'):
                db_smplx_path = osp.join(self.root_path, 'AGORA_' + split + '_SMPLX.json')
                db_smpl_path = osp.join(self.root_path, 'AGORA_' + split + '_SMPL.json')
                if osp.isfile(db_smplx_path):
                    db = COCO(db_smplx_path)
                elif osp.isfile(db_smpl_path):
                    db = COCO(db_smpl_path)
                else:
                    assert 0

                for aid in db.anns.keys():
                    ann = db.anns[aid]
                    img = db.loadImgs(ann['image_id'])[0]
                    person_id = ann['person_id']
                    bbox = np.array(ann['bbox']).reshape(4)
                    bbox = set_aspect_ratio(bbox, self.out_shape[1]/self.out_shape[0])

                    save_path = osp.join(self.root_path, 'images_3840x2160', img['file_name_3840x2160'].split('/')[-2] + '_crop')
                    os.makedirs(save_path, exist_ok=True)

                    self.datalist.append({
                                    'orig_img_path': osp.join(self.root_path, img['file_name_3840x2160']),
                                    'bbox': bbox,
                                    'save_img_path': osp.join(save_path, img['file_name_3840x2160'].split('/')[-1][:-4] + '_person_id_' + str(person_id) + '.png'),
                                    'save_json_path': osp.join(save_path, img['file_name_3840x2160'].split('/')[-1][:-4] + '_person_id_' + str(person_id) + '.json')
                                    })

            else:
                with open(osp.join(self.root_path, 'AGORA_test_bbox.json')) as f:
                    db = json.load(f)
                for filename in db.keys():
                    person_num = len(db[filename])
                    for person_id in range(person_num):
                        bbox = np.array(db[filename][person_id]['bbox']).reshape(4)
                        bbox = set_aspect_ratio(bbox, self.out_shape[1]/self.out_shape[0])

                        save_path = osp.join(self.root_path, 'images_3840x2160', 'test_crop')
                        os.makedirs(save_path, exist_ok=True)

                        self.datalist.append({
                                        'orig_img_path': osp.join(self.root_path, 'images_3840x2160', 'test', filename),
                                        'bbox': bbox,
                                        'save_img_path': osp.join(save_path, filename.split('/')[-1][:-4] + '_person_id_' + str(person_id) + '.png'),
                                        'save_json_path': osp.join(save_path, filename.split('/')[-1][:-4] + '_person_id_' + str(person_id) + '.json')
                                        })

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        orig_img_path, bbox, save_img_path, save_json_path = data['orig_img_path'], data['bbox'], data['save_img_path'], data['save_json_path']

        img = cv2.imread(orig_img_path)
        img, img2bb_trans, bb2img_trans = get_patch_img(img, bbox, self.out_shape)
        
        cv2.imwrite(save_img_path, img)
        with open(save_json_path, 'w') as f:
            json.dump({'bbox_orig': bbox.tolist(), 'affine_mat': img2bb_trans.tolist(), 'resized_height': self.out_shape[0], 'resized_width': self.out_shape[1]}, f)

        return 1


args = parse_args()
dataset = AGORA(args.dataset_path, int(args.out_height), int(args.out_width))
batch_size = 128
num_workers = 32
batch_generator = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
for _ in tqdm(batch_generator):
    pass
