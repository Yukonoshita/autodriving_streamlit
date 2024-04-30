import argparse

import cv2
import numpy as np
import torch

import utils


def str2bool(v):
    #print(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser()
# GPU parameters
parser.add_argument('--no_gpu', default=False, help='True for CPU')
# Saving, and loading parameters
parser.add_argument('--save_name', type=str, default='./results/results_tmp',
                    help='save the generated with certain epoch')
parser.add_argument('--load_name', type=str, default='efficientderain/models/v4_rain1400.pt',
                    help='load the pre-trained model with certain epoch')
# parser.add_argument('--load_name', type = str, default = './models/KPN_single_image_epoch120_bs16_mu0_sigma30.pth', help = 'load the pre-trained model with certain epoch')
parser.add_argument('--test_batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
# Initialization parameters
parser.add_argument('--color', type=str2bool, default=True, help='input type')
parser.add_argument('--burst_length', type=int, default=1, help='number of photos used in burst setting')
parser.add_argument('--blind_est', type=str2bool, default=True, help='variance map')
parser.add_argument('--kernel_size', type=list, default=[3], help='kernel size')
parser.add_argument('--sep_conv', type=str2bool, default=False, help='simple output type')
parser.add_argument('--channel_att', type=str2bool, default=False, help='channel wise attention')
parser.add_argument('--spatial_att', type=str2bool, default=False, help='spatial wise attention')
parser.add_argument('--upMode', type=str, default='bilinear', help='upMode')
parser.add_argument('--core_bias', type=str2bool, default=False, help='core_bias')
parser.add_argument('--init_type', type=str, default='xavier', help='initialization type of generator')
parser.add_argument('--init_gain', type=float, default=0.02, help='initialization gain of generator')
# Dataset parameters
parser.add_argument('--baseroot', type=str, default='./datasets/SPA/testing', help='images baseroot')
parser.add_argument('--crop', type=str2bool, default=False, help='whether to crop input images')
parser.add_argument('--crop_size', type=int, default=512, help='single patch size')
parser.add_argument('--geometry_aug', type=str2bool, default=False, help='geometry augmentation (scaling)')
parser.add_argument('--angle_aug', type=str2bool, default=False, help='geometry augmentation (rotation, flipping)')
parser.add_argument('--scale_min', type=float, default=1, help='min scaling factor')
parser.add_argument('--scale_max', type=float, default=1, help='max scaling factor')
parser.add_argument('--add_noise', type=str2bool, default=False, help='whether to add noise to input images')
parser.add_argument('--mu', type=int, default=0, help='Gaussian noise mean')
parser.add_argument('--sigma', type=int, default=30, help='Gaussian noise variance: 30 | 50 | 70')
opt = parser.parse_args()

generator = utils.create_generator(opt)


def derain(img_rainy):
    img_rainy = np.asarray(img_rainy)

    height = img_rainy.shape[0]
    width = img_rainy.shape[1]
    height_origin = height
    width_origin = width
    if height % 16 != 0:
        height = ((height // 16) + 1) * 16
    if width % 16 != 0:
        width = ((width // 16) + 1) * 16
    img_rainy = cv2.resize(img_rainy, (width, height))

    img_rainy = cv2.cvtColor(img_rainy, cv2.COLOR_BGR2RGB)

    img_rainy = img_rainy.astype(np.float32)
    img_rainy = img_rainy / 255.0
    img_rainy = torch.from_numpy(img_rainy.transpose(2, 0, 1)).contiguous()
    img_rainy = img_rainy.unsqueeze(0)
    img_rainy = img_rainy

    with torch.no_grad():
        img = generator(img_rainy, img_rainy)

    img = img * 255.0
    img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    img_copy = np.clip(img_copy, 0, 255)
    img_copy = img_copy.astype(np.uint8)[0, :, :, :]
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    if (height != -1) and (width != -1):
        img_copy = cv2.resize(img_copy, (width_origin, height_origin))

    return img_copy


if __name__ == '__main__':
    img_path = '../../data/16484_1_.png'
    img_rainy = cv2.imread(img_path)
    res = derain(img_rainy)
    cv2.imshow('res', res)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()