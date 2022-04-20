import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from pycocotools.coco import COCO

sys.path.insert(0, osp.join('..', '..', 'main'))
sys.path.insert(0, osp.join('..', '..', 'data'))
sys.path.insert(0, osp.join('..', '..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import process_bbox, generate_patch_image
from utils.human_models import smpl, smpl_x, mano, flame
from utils.vis import render_mesh, save_obj
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

args = parse_args()
cfg.set_args(args.gpu_ids, 'body')
cudnn.benchmark = True

# snapshot load
model_path = './snapshot_6_body.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
img_path = 'input_body.png'
original_img = cv2.imread(img_path)
original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox
bbox = [193, 120, 516-193, 395-120] # xmin, ymin, width, height
bbox = process_bbox(bbox, original_img_width, original_img_height)
img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
img = transform(img.astype(np.float32))/255
img = img.cuda()[None,:,:,:]
    
# forward
inputs = {'img': img}
targets = {}
meta_info = {}
with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')
mesh = out['smpl_mesh_cam'].detach().cpu().numpy()[0]

# save mesh
save_obj(mesh, smpl.face, 'output_body.obj')

# render mesh
vis_img = img.cpu().numpy()[0].transpose(1,2,0).copy() * 255
rendered_img = render_mesh(vis_img, mesh, smpl.face, {'focal': cfg.focal, 'princpt': cfg.princpt})
cv2.imwrite('render_cropped_img_body.jpg', rendered_img)

vis_img = original_img.copy()
focal = [cfg.focal[0] / cfg.input_img_shape[1] * bbox[2], cfg.focal[1] / cfg.input_img_shape[0] * bbox[3]]
princpt = [cfg.princpt[0] / cfg.input_img_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_img_shape[0] * bbox[3] + bbox[1]]
rendered_img = render_mesh(vis_img, mesh, smpl.face, {'focal': focal, 'princpt': princpt})
cv2.imwrite('render_original_img_body.jpg', rendered_img)

# save SMPL parameters
smpl_pose = out['smpl_pose'].detach().cpu().numpy()[0]; smpl_shape = out['smpl_shape'].detach().cpu().numpy()[0]; 
with open('smpl_param.json', 'w') as f:
    json.dump({'pose': smpl_pose.reshape(-1).tolist(), 'shape': smpl_shape.reshape(-1).tolist()}, f)

