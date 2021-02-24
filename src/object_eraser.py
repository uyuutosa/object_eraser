import torch
from numpy import *
import numpy as np
from easydict import EasyDict
import cv2
from network import GatedGenerator
from torchvision.transforms import ToTensor
import PIL.Image as I

class ObjectEraser:
    def __init__(self,
                 model_path='pretrained_model/deepfillv2_WGAN_G_epoch40_batchsize4.pth',
                 size=(512,512),
                 ):
        opt = EasyDict()
        opt.in_channels = 4
        opt.out_channels = 3
        opt.latent_channels = 48
        opt.pad_type = 'zero'
        opt.norm = 'none'
        opt.activation = 'elu'

        self.net = GatedGenerator(opt)
        self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.net.context_attention.use_cuda = False
        self.size = size

    def _resize(self, img):
        return cv2.resize(img,self.size)

    def _generate_mask(self, img):
        hsv_color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = ((hsv_color[..., 0] < 150) & (hsv_color[..., 0] > 100))[..., None] * ones(3)
        mask = cv2.dilate(mask, (9, 9), iterations=9)
        return where(mask, 255, 0).clip(0, 255).astype(uint8)

    def _load_img_and_mask(self, img_path, mask_path=None):

        if isinstance(img_path, str):
            img = cv2.imread(img_path)
        else:
            img = img_path # consider img_path as np.ndarray
        img = self._resize(img)

        if mask_path is None:
            mask = self._generate_mask(img)
        else:
            mask = cv2.imread(mask_path)
        mask = self._resize(mask)[:, :, 0]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()[None]
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous()[None]
        return img, mask

    def erase(self, img_path, mask_path=None):
        img, mask = self._load_img_and_mask(img_path, mask_path)
        with torch.no_grad():
            first_out, second_out = self.net(img, mask)
        result = img * (1 - mask) + second_out * mask
        ret = (result.detach().numpy() * 255).clip(0, 255).astype(uint8)[0].transpose(1,2,0)[...,::-1]
        return ret, (mask.detach().cpu().numpy()[0,0] * 255).clip(0,255).astype(uint8)
