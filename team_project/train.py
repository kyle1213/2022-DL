import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import rgbFSRCNN, FSRCNN, groupedFSRCNN, FSRCNN_x
from datasets import TrainDataset, EvalDataset, rgbTrainDataset, rgbEvalDataset
from utils import AverageMeter, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True, default='D:/datasets/SR task/train_h5.h5')
    parser.add_argument('--eval-file', type=str, required=True, default='D:/datasets/SR task/test_h5.h5')
    parser.add_argument('--outputs-dir', type=str, required=True, default='D:/User_Data/Desktop/github/2022-DL/team_project/models')
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--model', type=str, required=True, default='ySRCNN')
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    if args.model == 'FSRCNN':
        model = FSRCNN(scale_factor=args.scale).to(device)
    elif args.model == 'SRCNN_x':
        model = FSRCNN_x(scale_factor=args.scale).to(device)
    elif args.model == 'rgbSRCNN':
        model = rgbFSRCNN(scale_factor=args.scale).to(device)
    elif args.model == 'groupedSRCNN':
        model = groupedFSRCNN(scale_factor=args.scale).to(device)
    else:
        raise Exception("WRONG MODEL NAME")

      ###
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    if args.model == 'FSRCNN' or args.model == 'SRCNN_x':
        train_dataset = TrainDataset(args.train_file)  ###
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
        eval_dataset = EvalDataset(args.eval_file)  ###
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    elif args.model == 'rgbSRCNN' or args.model == 'groupedSRCNN':
        train_dataset = rgbTrainDataset(args.train_file)  ###
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
        eval_dataset = rgbEvalDataset(args.eval_file)  ###
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    else:
        raise Exception("WRONG MODEL NAME")

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

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

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
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
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
