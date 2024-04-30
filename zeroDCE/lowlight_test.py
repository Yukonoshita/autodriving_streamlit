import glob
import os
import time

import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image

from . import model


def lowlight(data_lowlight):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    scale_factor = 12

    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0)

    DCE_net = model.enhance_net_nopool(scale_factor)
    DCE_net.load_state_dict(torch.load('zeroDCE/Epoch99.pth', map_location=torch.device('cpu')))
    enhanced_image, params_maps = DCE_net(data_lowlight)

    # path = './zero_enhanced_images/'
    image = torchvision.utils.make_grid(enhanced_image).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).numpy()
    # torchvision.utils.save_image(enhanced_image, path)

    return image


if __name__ == '__main__':

    with torch.no_grad():

        filePath = 'data/test_data/'
        file_list = os.listdir(filePath)
        sum_time = 0
        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*")
            for image in test_list:
                print(image)
                sum_time = sum_time + lowlight(image)

        print(sum_time)
