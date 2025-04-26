import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import sys, os, argparse

sys.path.append('src/dataset')
sys.path.append('src/model')
import resnet
import cv2
import glob
from PIL import Image

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', help='Path of video.',
                    default='', type=str)

    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    outdir = 'data/affect_features'
    snapshot_path = 'models/an_resnet50_s_epoch_10.pkl'

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    model = resnet.resnet50(pretrained=False, num_classes=7)
        
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    transformations = transforms.Compose([transforms.CenterCrop(256),
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.cuda()

    print('Ready to test network.')

    print('Testing.')

    videos = glob.glob(args.path)
    for vid_path in videos:
        vidnum = vid_path.split('/')[-1].split('_')[0]
        outpath = os.path.join(outdir, vidnum + '.npy')
        print(outpath)
        
        if os.path.exists(outpath):
            continue
        
        feature_list = []
       
        image_list = sorted(glob.glob(os.path.join(vid_path, '*.png')), key=lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-1]))
        
        for i, img_path in enumerate(image_list):
            img_num = int(img_path.split('/')[-1].split('.')[0].split('-')[-1])
            
            img = Image.open(img_path)
            img = img.convert('RGB').resize((256, 256))
            img = transformations(img)
            img = img[None, :]
            img = img.cuda()
            exp_output, features = model(img)
            print(i+1, img_num)
            assert((i+1) == img_num)
            feature_list.append(features.cpu().numpy())
        
        vid_features = np.array(feature_list)
        np.save(outpath, vid_features)
