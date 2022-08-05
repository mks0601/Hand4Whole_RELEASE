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

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.human_models import smpl_x
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
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# snapshot load
model_path = './snapshot_6.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
img_path = './input.png'
original_img = load_img(img_path)
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
mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

# save mesh
save_obj(mesh, smpl_x.face, 'output.obj')

# render mesh
vis_img = img.cpu().numpy()[0].transpose(1,2,0).copy() * 255
focal = [cfg.focal[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1], cfg.focal[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]]
princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1], cfg.princpt[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]]
rendered_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
cv2.imwrite('render_cropped_img.jpg', rendered_img)

vis_img = original_img.copy()
focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
rendered_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
cv2.imwrite('render_original_img.jpg', rendered_img)

# save SMPL-X parameters
root_pose = out['smplx_root_pose'].detach().cpu().numpy()[0]
body_pose = out['smplx_body_pose'].detach().cpu().numpy()[0] 
lhand_pose = out['smplx_lhand_pose'].detach().cpu().numpy()[0] 
rhand_pose = out['smplx_rhand_pose'].detach().cpu().numpy()[0] 
jaw_pose = out['smplx_jaw_pose'].detach().cpu().numpy()[0] 
shape = out['smplx_shape'].detach().cpu().numpy()[0]
expr = out['smplx_expr'].detach().cpu().numpy()[0] 
with open('smplx_param.json', 'w') as f:
    json.dump({'root_pose': root_pose.reshape(-1).tolist(), \
            'body_pose': body_pose.reshape(-1).tolist(), \
            'lhand_pose': lhand_pose.reshape(-1).tolist(), \
            'rhand_pose': rhand_pose.reshape(-1).tolist(), \
            'jaw_pose': jaw_pose.reshape(-1).tolist(), \
            'shape': shape.reshape(-1).tolist(), \
            'expr': expr.reshape(-1).tolist()}, f)

