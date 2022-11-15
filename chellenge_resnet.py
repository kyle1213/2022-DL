import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import _csv
import torch.optim as optim
from torchsummary import summary as summary_


cuda = torch.device('cuda')

train_data_npy_path = "D:/datasets/2022_DL_challenge/2022ajouit2/trainset.npy"
train_label_npy_path = "D:/datasets/2022_DL_challenge/2022ajouit2/trainlabel.npy"
test_data_npy_path = "D:/datasets/2022_DL_challenge/2022ajouit2/testset.npy"

x = np.load(train_data_npy_path)
y = np.load(train_label_npy_path)
x_test = np.load(test_data_npy_path)

#x = np.swapaxes(x, 1, 2)
x = np.swapaxes(x, 1, 3)
#x_test = np.swapaxes(x_test, 1, 2)
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


x_train = x[:int(0.8 * len(x))]
y_train = y[:int(0.8 * len(y))]

x_val = x[int(0.8 * len(x)):]
y_val = y[int(0.8 * len(x)):]

print("x_train shape: ", np.shape(x_train))
print("y_train shape: ", np.shape(y_train))
print("x_val shape: ", np.shape(x_val))
print("y_val shape: ", np.shape(y_val))
print("x_test shape: ", np.shape(x_test))

train_dataset = CustomDataset(x_train, y_train, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
]))
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = CustomDataset(x_val, y_val, transform=transforms.Compose([transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]))
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

test_dataset = testCustomDataset(x_test, transform=transforms.Compose([transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]))
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def train_loop(dataloader, model, loss_fn, optimizer):
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


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(cuda)
            y = y.to(cuda)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")





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
            nn.Linear(2*2*32, 100)
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

        #x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x


model = NeuralNetwork().to(cuda)

print('mobilenetv3:\n', model)
print('Total params: %', (sum(p.numel() for p in model.parameters())))

print(summary_(model,(3, 32, 32),batch_size=1))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=4e-5, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=0.001)
epochs = 150

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(val_dataloader, model, loss_fn)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    scheduler.step()

    if t%15 == 1:
      path2 = 'D:/User_Data/Desktop/github/2022-DL/challenge/model_res' + str(t) +'.pt'
      torch.save(model.state_dict(), path2)
print("Done!")

path2 = 'D:/User_Data/Desktop/github/2022-DL/challenge/model_res.pt'
torch.save(model.state_dict(), path2)