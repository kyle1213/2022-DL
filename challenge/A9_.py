import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary as summary_
import torch.optim as optim


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class DropBlock3D(DropBlock2D):
    r"""Randomly zeroes 3D spatial blocks of the input tensor.
    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, None, :, :, :],
                                  kernel_size=(self.block_size, self.block_size, self.block_size),
                                  stride=(1, 1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 3)


class SGDRScheduler(nn.Module):
    global_epoch = 0
    all_epoch = 0
    cur_drop_prob = 0.

    def __init__(self, dropblock):
        super(SGDRScheduler, self).__init__()
        self.dropblock = dropblock
        self.drop_values = 0.

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        # self.dropblock.drop_prob = np.abs((0 + 0.5 * 0.1 * (1 + np.cos(np.pi * SGDRScheduler.global_epoch / SGDRScheduler.all_epoch)))-0.1)
        # SGDRScheduler.cur_drop_prob = self.dropblock.drop_prob
        ix = np.log2(self.global_epoch / 10 + 1).astype(np.int)
        T_cur = self.global_epoch - 10 * (2 ** (ix) - 1)
        T_i = (10 * 2 ** ix)
        self.dropblock.drop_prob = np.abs((0 + 0.5 * 0.1 * (1 + np.cos(np.pi * T_cur / T_i))) - 0.1)
        SGDRScheduler.cur_drop_prob = self.dropblock.drop_prob


class LinearScheduler(nn.Module):
    global_epoch = 0
    num_epochs = 0

    def __init__(self, dropblock, start_value=0., stop_value=0.1):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=self.num_epochs)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        self.dropblock.drop_prob = self.drop_values[self.global_epoch]


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


import math
import time


def adjust_lr(optimizer, epoch, eta_max=0.1, eta_min=0.):
    cur_lr = 0.
    if lr_type == 'SGDR':
        i = int(math.log2(epoch / 10 + 1))
        T_cur = epoch - 10 * (2 ** (i) - 1)
        T_i = (10 * 2 ** i)

        cur_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))

    for param_group in optimizer.param_groups:
        if epoch >= 0:
            cur_lr = 0.001
        param_group['lr'] = cur_lr
    return cur_lr


drop_scheduler = SGDRScheduler
lr_type = 'SGDR'

cuda = torch.device('cuda')

train_data_npy_path = "D:/datasets/2022_DL_challenge/2022ajouit2/trainset.npy"
train_label_npy_path = "D:/datasets/2022_DL_challenge/2022ajouit2/trainlabel.npy"
test_data_npy_path = "D:/datasets/2022_DL_challenge/2022ajouit2/testset.npy"

x = np.load(train_data_npy_path)
y = np.load(train_label_npy_path)
x_test = np.load(test_data_npy_path)

x = np.swapaxes(x, 1, 2)
x = np.swapaxes(x, 1, 3)
x_test = np.swapaxes(x_test, 1, 2)
x_test = np.swapaxes(x_test, 1, 3)


class CustomDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x_data = x
        self.y_data = y
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = self.y_data[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


x_train = x
y_train = y

print("x_train shape: ", np.shape(x_train))
print("y_train shape: ", np.shape(y_train))
print("x_test shape: ", np.shape(x_test))

batch_size = 64
train_dataset = CustomDataset(x_train, y_train, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
]))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=2, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.batch1 = nn.BatchNorm2d(16)
        self.batch2 = nn.BatchNorm2d(32)
        self.batch3 = nn.BatchNorm2d(32)
        self.batch4 = nn.BatchNorm2d(32)
        self.batch5 = nn.BatchNorm2d(32)
        self.batch6 = nn.BatchNorm2d(32)

        # 224 224 3
        self.conv1_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2_layer = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv3_layer = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv4_layer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv5_layer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv6_layer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 1 1 512
        self.fc_layer = nn.Sequential(
            nn.Linear(2 * 2 * 32, 100)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1_layer(x)
        shortcut = self.conv1(x)
        shortcut = self.batch1(shortcut)
        shortcut = nn.ReLU()(shortcut)

        x = self.conv2_layer(x)
        x = nn.ReLU()(x + shortcut)
        shortcut = self.conv2(x)
        shortcut = self.batch2(shortcut)
        shortcut = nn.ReLU()(shortcut)

        x = self.conv3_layer(x)
        x = nn.ReLU()(x + shortcut)
        shortcut = self.conv3(x)
        shortcut = self.batch3(shortcut)
        shortcut = nn.ReLU()(shortcut)

        x = self.conv4_layer(x)
        x = nn.ReLU()(x + shortcut)
        shortcut = self.conv4(x)
        shortcut = self.batch4(shortcut)
        shortcut = nn.ReLU()(shortcut)

        x = self.conv5_layer(x)
        x = nn.ReLU()(x + shortcut)
        shortcut = self.conv5(x)
        shortcut = self.batch5(shortcut)
        shortcut = nn.ReLU()(shortcut)

        x = self.conv6_layer(x)
        x = nn.ReLU()(x + shortcut)

        # x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x


__all__ = ['MobileNetV3', 'mobilenetv3']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, n_class=100, input_size=32, dropout=0, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 96

        if mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'RE', 2],
                [3, 64, 24, False, 'RE', 1],
                [3, 88, 40, False, 'RE', 1],
                [3, 96, 40, True, 'HS', 2],
                [3, 120, 48, True, 'HS', 1],
                [3, 224, 96, True, 'HS', 1]
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),  # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        model.load_state_dict(state_dict, strict=True)
        # raise NotImplementedError
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import math


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, dilation=1):
        super(BasicConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation=dilation, groups=groups, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class _SMG(nn.Module):
    def __init__(self, in_channels, growth_rate,
                 bn_size=10000, groups=4, reduction_factor=2, forget_factor=2):
        super(_SMG, self).__init__()
        self.in_channels = in_channels
        self.reduction_factor = reduction_factor
        self.forget_factor = forget_factor
        self.growth_rate = growth_rate
        self.conv1_1x1 = BasicConv(in_channels, bn_size * growth_rate, kernel_size=1, stride=1)
        self.conv2_3x3 = BasicConv(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1,
                                   padding=1, groups=groups)

        # Mobile
        self.conv_3x3 = BasicConv(growth_rate, growth_rate, kernel_size=3,
                                  stride=1, padding=1, groups=growth_rate, )
        self.conv_5x5 = BasicConv(growth_rate, growth_rate, kernel_size=3,
                                  stride=1, padding=2, groups=growth_rate, dilation=2)

        # GTSK layers
        self.global_context3x3 = nn.Conv2d(growth_rate, 1, kernel_size=1)
        self.global_context5x5 = nn.Conv2d(growth_rate, 1, kernel_size=1)

        self.fcall = nn.Conv2d(2 * growth_rate, 2 * growth_rate // self.reduction_factor, kernel_size=1)
        self.bn_attention = nn.BatchNorm1d(2 * growth_rate // self.reduction_factor)
        self.fc3x3 = nn.Conv2d(2 * growth_rate // self.reduction_factor, growth_rate, kernel_size=1)
        self.fc5x5 = nn.Conv2d(2 * growth_rate // self.reduction_factor, growth_rate, kernel_size=1)

        # SE layers
        self.global_forget_context = nn.Conv2d(growth_rate, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_forget = nn.BatchNorm1d(growth_rate // self.forget_factor)
        self.fc1 = nn.Conv2d(growth_rate, growth_rate // self.forget_factor, kernel_size=1)
        self.fc2 = nn.Conv2d(growth_rate // self.forget_factor, growth_rate, kernel_size=1)

    def forward(self, x):
        x_dense = x
        x = self.conv1_1x1(x)
        x = self.conv2_3x3(x)

        H = W = x.size(-1)
        C = x.size(1)
        x_shortcut = x

        forget_context_weight = self.global_forget_context(x_shortcut)
        forget_context_weight = torch.flatten(forget_context_weight, start_dim=1)
        forget_context_weight = F.softmax(forget_context_weight, 1).reshape(-1, 1, H, W)
        x_shortcut_weight = self.global_pool(x_shortcut * forget_context_weight) * H * W

        x_shortcut_weight = \
            torch.tanh(self.bn_forget(torch.flatten(self.fc1(x_shortcut_weight), start_dim=1))) \
                .reshape(-1, C // self.forget_factor, 1, 1)
        x_shortcut_weight = torch.sigmoid(self.fc2(x_shortcut_weight))

        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        context_weight_3x3 = \
            F.softmax(torch.flatten(self.global_context3x3(x_3x3), start_dim=1), 1).reshape(-1, 1, H, W)
        context_weight_5x5 = \
            F.softmax(torch.flatten(self.global_context5x5(x_5x5), start_dim=1), 1).reshape(-1, 1, H, W)
        x_3x3 = self.global_pool(x_3x3 * context_weight_3x3) * H * W
        x_5x5 = self.global_pool(x_5x5 * context_weight_5x5) * H * W
        x_concat = torch.cat([x_3x3, x_5x5], 1)
        attention = torch.tanh(self.bn_attention(torch.flatten(self.fcall(x_concat), start_dim=1))) \
            .reshape(-1, 2 * C // self.reduction_factor, 1, 1)
        weight_3x3 = torch.unsqueeze(torch.flatten(self.fc3x3(attention), start_dim=1), 1)
        weight_5x5 = torch.unsqueeze(torch.flatten(self.fc5x5(attention), start_dim=1), 1)
        weight_all = F.softmax(torch.cat([weight_3x3, weight_5x5], 1), 1)
        weight_3x3, weight_5x5 = weight_all[:, 0, :].reshape(-1, C, 1, 1), weight_all[:, 1, :].reshape(-1, C, 1, 1)
        new_x = weight_3x3 * x_3x3 + weight_5x5 * x_5x5
        x = x_shortcut * x_shortcut_weight + new_x

        return torch.cat([x_dense, x], 1)


class _HybridBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, groups):
        super(_HybridBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('SMG%d' % (i + 1),
                            _SMG(in_channels + growth_rate * i,
                                 growth_rate, bn_size, groups))


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels, forget_factor=4, reduction_factor=4):
        super(_Transition, self).__init__()
        self.in_channels = in_channels
        self.forget_factor = forget_factor
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.reduce_channels = (in_channels - out_channels) // 2
        self.conv1_1x1 = BasicConv(in_channels, in_channels - self.reduce_channels, kernel_size=1, stride=1)
        self.conv2_3x3 = BasicConv(in_channels - self.reduce_channels, out_channels, kernel_size=3, stride=2,
                                   padding=1, groups=1)
        # Mobile
        # Mobile
        self.conv_3x3 = BasicConv(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1, groups=out_channels)
        self.conv_5x5 = BasicConv(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=2, dilation=2, groups=out_channels)

        # GTSK layers
        self.global_context3x3 = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.global_context5x5 = nn.Conv2d(out_channels, 1, kernel_size=1)

        self.fcall = nn.Conv2d(2 * out_channels, 2 * out_channels // self.reduction_factor, kernel_size=1)
        self.bn_attention = nn.BatchNorm1d(2 * out_channels // self.reduction_factor)
        self.fc3x3 = nn.Conv2d(2 * out_channels // self.reduction_factor, out_channels, kernel_size=1)
        self.fc5x5 = nn.Conv2d(2 * out_channels // self.reduction_factor, out_channels, kernel_size=1)

        # SE layers
        self.global_forget_context = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_forget = nn.BatchNorm1d(out_channels // self.forget_factor)
        self.fc1 = nn.Conv2d(out_channels, out_channels // self.forget_factor, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channels // self.forget_factor, out_channels, kernel_size=1)
        self.dropblock = SGDRScheduler(DropBlock2D(drop_prob=0, block_size=2))

    def forward(self, x):
        self.dropblock.step()
        x = self.conv1_1x1(x)
        x = self.conv2_3x3(x)

        H = W = x.size(-1)
        C = x.size(1)
        x_shortcut = x

        forget_context_weight = self.global_forget_context(x_shortcut)
        forget_context_weight = torch.flatten(forget_context_weight, start_dim=1)
        forget_context_weight = F.softmax(forget_context_weight, 1)
        forget_context_weight = forget_context_weight.reshape(-1, 1, H, W)
        x_shortcut_weight = self.global_pool(x_shortcut * forget_context_weight) * H * W

        x_shortcut_weight = \
            torch.tanh(self.bn_forget(torch.flatten(self.fc1(x_shortcut_weight), start_dim=1))) \
                .reshape(-1, C // self.forget_factor, 1, 1)
        x_shortcut_weight = torch.sigmoid(self.fc2(x_shortcut_weight))

        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        context_weight_3x3 = \
            F.softmax(torch.flatten(self.global_context3x3(x_3x3), start_dim=1), 1).reshape(-1, 1, H, W)
        context_weight_5x5 = \
            F.softmax(torch.flatten(self.global_context5x5(x_5x5), start_dim=1), 1).reshape(-1, 1, H, W)
        x_3x3 = self.global_pool(x_3x3 * context_weight_3x3) * H * W
        x_5x5 = self.global_pool(x_5x5 * context_weight_5x5) * H * W
        x_concat = torch.cat([x_3x3, x_5x5], 1)
        attention = torch.tanh(self.bn_attention(torch.flatten(self.fcall(x_concat), start_dim=1))) \
            .reshape(-1, 2 * C // self.reduction_factor, 1, 1)
        weight_3x3 = torch.unsqueeze(torch.flatten(self.fc3x3(attention), start_dim=1), 1)
        weight_5x5 = torch.unsqueeze(torch.flatten(self.fc5x5(attention), start_dim=1), 1)
        weight_all = F.softmax(torch.cat([weight_3x3, weight_5x5], 1), 1)
        weight_3x3, weight_5x5 = weight_all[:, 0, :].reshape(-1, C, 1, 1), weight_all[:, 1, :].reshape(-1, C, 1, 1)
        new_x = weight_3x3 * x_3x3 + weight_5x5 * x_5x5

        x = x_shortcut * x_shortcut_weight + new_x

        return self.dropblock(x)

        # return x


class HCGNet(nn.Module):
    def __init__(self, growth_rate=(8, 16, 32), block_config=(6, 12, 24, 16),
                 bn_size=4, theta=0.5, num_classes=10, groups=6):
        super(HCGNet, self).__init__()
        num_init_feature = 2 * growth_rate[0]

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_feature,
                                kernel_size=3, stride=1,
                                padding=1, bias=False)),
        ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('HybridBlock%d' % (i + 1),
                                     _HybridBlock(num_layers, num_feature, bn_size, growth_rate[i], groups))
            num_feature = num_feature + growth_rate[i] * num_layers
            if i != len(block_config) - 1:
                self.features.add_module('Transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.classifier = nn.Linear(num_feature, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = F.adaptive_avg_pool2d(F.relu(features), (1, 1))
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


def HCGNet_A1(num_classes=100):
    return HCGNet(growth_rate=(12, 24, 36), block_config=(2, 2, 2), num_classes=num_classes)


def HCGNet_A2(num_classes=100):
    return HCGNet(growth_rate=(4, 8, 16), block_config=(6, 6, 6), num_classes=num_classes)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(cuda)
        y = y.to(cuda)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 64 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(cuda)
            y = y.to(cuda)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for param_group in optimizer.param_groups:
        lr = param_group['lr']  # lr = adjust_lr(optimizer, epoch)
    drop_scheduler.global_epoch = epoch
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        iter_start_time = time.time()
        inputs, targets = inputs.to(cuda), targets.to(cuda)

        random_num = np.float64(np.random.rand(1))

        if (random_num >= 0.7):
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, 1.)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        elif (random_num >= 0.4):

            lam = np.random.beta(1., 1.)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            targets_a = targets
            targets_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (1. - lam)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        iter_start_time = time.time()
    print('Epoch:{0}\t lr:{1:.6f}\t duration:{2:.3f}\ttrain_acc:{3:.2f}\ttrain_loss:{4:.6f}'
          .format(epoch, lr, time.time() - start_time, 100. * correct / total, train_loss / (len(train_dataloader))))

    path2 = 'D:/User_Data/Desktop/github/2022-DL/challenge/A7_model/model' + str(epoch) + '.pt'
    torch.save(model.state_dict(), path2)


def HCGNet_A8(num_classes=100):
    return HCGNet(growth_rate=(2, 4, 8), block_config=(8, 10, 12), num_classes=num_classes, bn_size=5, groups=2)


model = HCGNet_A8().to(cuda)
summary_(model, (3, 32, 32), batch_size=1)

#path2 = 'D:/User_Data/Desktop/github/2022-DL/challenge/A7_model/model147.pt'
#model.load_state_dict(torch.load(path2))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

epochs = 500
for i in range(epochs):
    train(i)

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
for i in range(500, 1000):
    train(i)

optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
for i in range(1000, 1000 + 500):
    train(i)
print("Done!")


