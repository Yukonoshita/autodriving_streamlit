import argparse

import numpy as np
import torch
from skimage.io import imread
from torchvision import transforms
# from networks.MSBDN_DFF_v1_1 import Net

parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--isTest", type=bool, default=True, help="Test or not")
parser.add_argument('--dataset', type=str, default='SOTS', help='Path of the validation dataset')
parser.add_argument("--checkpoint", default="msbdn/model/model.pt", type=str,
                    help="Test on intermediate pkl (default: none)")
parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--name', type=str, default='MSBDN', help='filename of the training models')
parser.add_argument("--start", type=int, default=2, help="Activated gate module")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = parser.parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
str_ids = opt.gpu_ids.split(',')
torch.cuda.set_device(int(str_ids[0]))

# model = Net().cuda()
# model.load_state_dict(model.state_dict(), torch.load(opt.checkpoint, map_location=lambda storage, loc: storage))
# model.eval()

model = torch.load(opt.checkpoint, map_location=lambda storage, loc: storage)
# torch.save(model.state_dict(), 'model_state_dict.pkl')


def dehaze(input_image):
    # input_image = imread(os.path.join(self.input_dir, "%s" % name))
    input_image = np.asarray(input_image)
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.asarray(input_image, np.float32)
    input_image /= 255
    tensor = torch.from_numpy(input_image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    model.to(device)
    sr = model(tensor)

    try:
        sr = torch.clamp(sr, min=0, max=1)
    except:
        sr = sr[0]
        sr = torch.clamp(sr, min=0, max=1)

    resultSRDeblur = transforms.ToPILImage()(sr.cpu()[0])
    return resultSRDeblur


if __name__ == '__main__':
    image = imread('./SOTS/HR_hazy/00633.png')
    image = dehaze(image)
    image.save('./001.png')
    # torch.save(model.state_dict, 'model_state_dict.pkl')
