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

scale = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 1

# utils
def calc_patch_size(func):
    def wrapper(args):
        if args.scale == 2:
            args.patch_size = 10
        elif args.scale == 3:
            args.patch_size = 7
        elif args.scale == 4:
            args.patch_size = 6
        else:
            raise Exception('Scale Error', args.scale)
        return func(args)
    return wrapper


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


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

val_data = []
for path in val_path:
    imgs = os.listdir(path)
    for img in imgs:
        img_dir = path + '/' + img
        if '.png' in img:
            f = cv2.imread(img_dir)
            val_data.append(Image.fromarray(f))


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
    transforms.RandomHorizontalFlip(),
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
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),  # feature extraction
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]  # dim shrink
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])  # non-linear
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])  # dim expand
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)  # deconvolution(upsacale)

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
torch.load('./model/epoch_138.pth', model.state_dict())
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=True)

num_epochs = 200
best_psnr = 0.0
best_epoch = 0
best_weights = copy.deepcopy(model.state_dict())
for epoch in range(138, num_epochs):
    model.train()
    epoch_losses = AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))

        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))

    torch.save(model.state_dict(), os.path.join('D:/User_Data/Desktop/github/2022-DL/team_project/model', 'epoch_{}.pth'.format(epoch)))

    model.eval()
    epoch_psnr = AverageMeter()

    for data in val_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

    if epoch_psnr.avg > best_psnr:
        best_epoch = epoch
        best_psnr = epoch_psnr.avg
        best_weights = copy.deepcopy(model.state_dict())

print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
torch.save(best_weights, os.path.join('D:/User_Data/Desktop/github/2022-DL/team_project/model', 'best.pth'))