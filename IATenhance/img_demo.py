import argparse
import warnings

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Normalize

from model.IAT_main import IAT

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='demo_imgs/dayClip13--00008.jpg')
parser.add_argument('--normalize', type=bool, default=False)
parser.add_argument('--task', type=str, default='exposure', help='Choose from exposure or enhance')
config = parser.parse_args()

# Weights path
exposure_pretrain = r'IATenhance/best_Epoch_exposure.pth'
enhance_pretrain = r'best_Epoch_lol_v1.pth'

normalize_process = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

## Load Pre-train Weights
model = IAT()
if config.task == 'exposure':
    model.load_state_dict(torch.load(exposure_pretrain, map_location=torch.device('cpu')))
elif config.task == 'enhance':
    model.load_state_dict(torch.load(enhance_pretrain, map_location=torch.device('cpu')))
else:
    warnings.warn('Only could be exposure or enhance')
model.eval()


def exposure(img):
    img = (np.asarray(img) / 255.0)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    input = torch.from_numpy(img).float()
    input = input.permute(2, 0, 1).unsqueeze(0)
    if config.normalize:  # False
        input = normalize_process(input)

    ## Forward Network
    _, _, enhanced_img = model(input)

    return transforms.ToPILImage()(enhanced_img[0])
