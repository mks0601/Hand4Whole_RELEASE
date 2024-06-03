from glob import glob
import torch
import os.path as osp
import pickle
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, dest='dataset_path')
    parser.add_argument('--human_model', type=str, dest='human_model')
    args = parser.parse_args()

    assert args.dataset_path, "Please set dataset_path"
    assert args.human_model, "Please set human_model"
    return args

args = parse_args()
root_path = osp.join(args.dataset_path, args.human_model + '_gt')
folder_path_list = glob(osp.join(root_path, '*'))
for folder_path in folder_path_list:
    pkl_path_list = glob(osp.join(folder_path, '*.pkl'))
    for pkl_path in tqdm(pkl_path_list):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        for k in data.keys():
            if type(data[k]) is torch.Tensor:
                data[k] = data[k].cpu().detach().numpy()

        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)

