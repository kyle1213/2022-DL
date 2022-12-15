import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import os
import cv2
from PIL import Image
import PIL.Image as pil_image
import math
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt


scale = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 1


# utils
def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# data read
train_path = ["D:/datasets/SR task/train/DIV2K/DIV2K_train_HR", "D:/datasets/SR task/train/Flickr2K"]
val_path = ["D:/datasets/SR task/test/DIV2K_valid"]

train_data = []
for path in train_path:
    imgs = os.listdir(path)
    for img in imgs:
        img_dir = path + '/' + img
        if '.png' in img:
            f = cv2.imread(img_dir)
            train_data.append(Image.fromarray(f))
            break

val_data = []
for path in val_path:
    imgs = os.listdir(path)
    for img in imgs:
        img_dir = path + '/' + img
        if '.png' in img:
            f = cv2.imread(img_dir)
            val_data.append(Image.fromarray(f))
            break


# data loader
class srDataset(Dataset):
    def __init__(self, x, transforms=None):
        super(srDataset, self).__init__()
        self.x = x
        self.transforms = transforms

    def __getitem__(self, idx):
        self.hr = self.x[idx]
        if self.transforms:
            self.hr = self.transforms(self.hr)
        self.lr = self.hr.resize((int(self.hr.width // scale), int(self.hr.height // scale)),
                                 resample=pil_image.BICUBIC)

        self.lr, self.hr = transforms.ToTensor()(self.lr), transforms.ToTensor()(self.hr)

        return self.lr, self.hr

    def __len__(self):

        return len(self.x)


class EvalDataset(Dataset):
    def __init__(self, x, transforms=None):
        super(EvalDataset, self).__init__()
        self.hr = x
        self.transforms = transforms

    def __getitem__(self, idx):
        if self.transforms:
            self.hr = self.transforms(self.hr[idx])
        self.lr = self.hr.resize((torch.div(self.hr[1], scale, rounding_mode='floor'), torch.div(self.hr[2], scale, rounding_mode='floor')))

        return self.lr, self.hr

    def __len__(self):
        return len(self.hr)


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage()
])
train_dataset = srDataset(train_data, transforms=train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage()
])
val_dataset = srDataset(val_data, transforms=val_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)


# model
class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


# train
model = FSRCNN(scale_factor=scale).to(device)

torch.load('D:/User_Data/Desktop/github/2022-DL/team_project/model/fsrcnn_x2.pth', model.state_dict())

for data in train_dataloader:
    inputs, labels = data
    plt.subplot(1, 3, 1)
    plt.imshow(inputs.squeeze(0).permute(1, 2, 0))
    plt.subplot(1, 3, 2)
    plt.imshow(labels.squeeze(0).permute(1, 2, 0))
    inputs = inputs.to(device)
    labels = labels.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(inputs)
        plt.subplot(1, 3, 3)
        a = torch.Tensor.cpu(preds)*255
        a = nn.ReLU()(a)
        plt.imshow(a.squeeze(0).permute(1, 2, 0))
    plt.show()

    #plot pred
