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
cfg.set_args(args.gpu_ids, 'hand')
cudnn.benchmark = True

# snapshot load
model_path = './snapshot_12_hand.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
img_path = 'input_hand.png'
original_img = cv2.imread(img_path)
original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox (right hand)
rhand_bbox = [239, 218, 305-239, 295-218] # xmin, ymin, width, height
rhand_bbox = process_bbox(rhand_bbox, original_img_width, original_img_height)
rhand_img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, rhand_bbox, 1.0, 0.0, False, cfg.input_img_shape) 
rhand_img = transform(rhand_img.astype(np.float32))/255
rhand_img = rhand_img.cuda()[None,:,:,:]

# prepare bbox (left hand)
lhand_bbox = [306, 281, 368-306, 355-281] # xmin, ymin, width, height
lhand_bbox = process_bbox(lhand_bbox, original_img_width, original_img_height)
lhand_img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, lhand_bbox, 1.0, 0.0, True, cfg.input_img_shape) # flip to the right hand image
lhand_img = transform(lhand_img.astype(np.float32))/255
lhand_img = lhand_img.cuda()[None,:,:,:]

# forward
img = torch.cat((rhand_img, lhand_img))
inputs = {'img': img}
targets = {}
meta_info = {}
with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')
rhand_mesh = out['mano_mesh_cam'].detach().cpu().numpy()[0]
lhand_mesh = out['mano_mesh_cam'].detach().cpu().numpy()[1]
lhand_img = torch.flip(lhand_img, [3]) # flip back to the left hand image
lhand_mesh[:,0] *= -1 # flip back to the left hand mesh

# save mesh
save_obj(rhand_mesh, mano.face['right'], 'output_rhand.obj')
save_obj(lhand_mesh, mano.face['left'], 'output_lhand.obj')

# render mesh
vis_img = rhand_img.cpu().numpy()[0].transpose(1,2,0).copy() * 255
rendered_img = render_mesh(vis_img, rhand_mesh, mano.face['right'], {'focal': cfg.focal, 'princpt': cfg.princpt})
cv2.imwrite('render_cropped_img_rhand.jpg', rendered_img)
vis_img = lhand_img.cpu().numpy()[0].transpose(1,2,0).copy() * 255
rendered_img = render_mesh(vis_img, lhand_mesh, mano.face['left'], {'focal': cfg.focal, 'princpt': cfg.princpt})
cv2.imwrite('render_cropped_img_lhand.jpg', rendered_img)

vis_img = original_img.copy()
focal = [cfg.focal[0] / cfg.input_img_shape[1] * rhand_bbox[2], cfg.focal[1] / cfg.input_img_shape[0] * rhand_bbox[3]]
princpt = [cfg.princpt[0] / cfg.input_img_shape[1] * rhand_bbox[2] + rhand_bbox[0], cfg.princpt[1] / cfg.input_img_shape[0] * rhand_bbox[3] + rhand_bbox[1]]
rendered_img = render_mesh(vis_img, rhand_mesh, mano.face['right'], {'focal': focal, 'princpt': princpt})
cv2.imwrite('render_original_img_rhand.jpg', rendered_img)
vis_img = original_img.copy()
focal = [cfg.focal[0] / cfg.input_img_shape[1] * lhand_bbox[2], cfg.focal[1] / cfg.input_img_shape[0] * lhand_bbox[3]]
princpt = [cfg.princpt[0] / cfg.input_img_shape[1] * lhand_bbox[2] + lhand_bbox[0], cfg.princpt[1] / cfg.input_img_shape[0] * lhand_bbox[3] + lhand_bbox[1]]
rendered_img = render_mesh(vis_img, lhand_mesh, mano.face['left'], {'focal': focal, 'princpt': princpt})
cv2.imwrite('render_original_img_lhand.jpg', rendered_img)


# save MANO parameters
mano_pose = out['mano_pose'].detach().cpu().numpy(); mano_shape = out['mano_shape'].detach().cpu().numpy();
rmano_pose, rmano_shape = mano_pose[0], mano_shape[0]
with open('mano_param_rhand.json', 'w') as f:
    json.dump({'pose': rmano_pose.reshape(-1).tolist(), 'shape': rmano_shape.reshape(-1).tolist()}, f)
lmano_pose, lmano_shape = mano_pose[1], mano_shape[1]
lmano_pose = lmano_pose.reshape(-1,3)
lmano_pose[:,1:3] *= -1
with open('mano_param_lhand.json', 'w') as f:
    json.dump({'pose': lmano_pose.reshape(-1).tolist(), 'shape': lmano_shape.reshape(-1).tolist()}, f)


