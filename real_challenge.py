import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchsummary import summary as summary_


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


class testCustomDataset(Dataset):
  def __init__(self, x, transform=None):
    self.x_data = x
    self.transform = transform

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    x = torch.FloatTensor(self.x_data[idx])

    if self.transform:
      x = self.transform(x)
    return x


train_dataset = CustomDataset(x, y, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
]))
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = testCustomDataset(x_test, transform=transforms.Compose([transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]))
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def train_loop(dataloader, model, loss_fn, optimizer):
    mean_loss = []
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(cuda)
        y = y.to(cuda)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 64 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            mean_loss.append(loss)
    return mean_loss


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
    def __init__(self, channel, reduction=4):
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
            nlin_layer = nn.ReLU # or ReLU6
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

"""
v1
[
                # k, exp, c,  se,     nl,  s,
                [3, 32,  32,  True,  'RE', 2],
                [3, 32,  32,  False, 'RE', 1],
                [3, 32,  32,  False, 'RE', 1],
                [3, 32,  32,  True,  'HS', 2],
                [3, 32, 32,  True,  'HS', 1],
                [3, 32, 32,  True,  'HS', 1],
                [3, 64, 64,  True,  'HS', 2],
                [3, 64, 64,  True,  'HS', 1],
                [3, 64, 64,  True,  'HS', 1],
]
 + 256 at last(maybe)
"""
class MobileNetV3(nn.Module):
    def __init__(self, n_class=100, input_size=32, dropout=0.5, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 128

        if mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 24,  16,  False, 'RE', 1],
                [3, 24,  16,  False, 'RE', 1],
                [3, 32,  32,  True,  'HS', 2],
                [3, 40, 32,  True,  'HS', 1],
                [3, 48, 32,  True,  'HS', 1],
                [3, 48, 32,  True,  'HS', 1],
                [3, 56, 32,  True,  'HS', 1],
                [3, 64, 64,  True,  'HS', 2],
                [3, 80, 64,  True,  'HS', 1],
                [3, 96, 64,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(96 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
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


cuda = torch.device('cuda')
model = mobilenetv3().to(cuda)
#path2 = 'D:/User_Data/Desktop/github/2022-DL/mobile_net_real/model149.pt'
#model.load_state_dict(torch.load(path2))
print('mobilenetv3:\n', model)
print('Total params: %', (sum(p.numel() for p in model.parameters())))

print(summary_(model,(3, 32, 32),batch_size=1))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=4e-5, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=0.001)
epochs = 300

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    mean_l = train_loop(train_dataloader, model, loss_fn, optimizer)
    print(sum(mean_l)/50, sum(mean_l)/49)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    scheduler.step()
    path2 = 'D:/User_Data/Desktop/github/2022-DL/mobile_net_realv2/model' + str(t) +'.pt'
    torch.save(model.state_dict(), path2)
print("Done!")