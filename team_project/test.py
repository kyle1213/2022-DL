import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import FSRCNN, rgbFSRCNN, groupedFSRCNN, FSRCNN_x
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr, rgbpreprocess

import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    start = time.time()
    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    if args.model == 'FSRCNN' or args.model == 'SRCNN_x':
        lr, _ = preprocess(lr, device)
        hr, _ = preprocess(hr, device)
        _, ycbcr = preprocess(bicubic, device)

        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)
        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        psnr = calc_psnr(hr, preds)
        print('PSNR: {:.2f}'.format(psnr))

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(args.image_file.replace('.', args.model + '_x{}.'.format(args.scale)))
    elif args.model == 'rgbSRCNN' or args.model == 'groupedSRCNN':
        lr = rgbpreprocess(lr, device)
        hr = rgbpreprocess(hr, device)

        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)
        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        psnr = calc_psnr(hr, preds)
        print('PSNR: {:.2f}'.format(psnr))
        preds = preds.mul(255.0).cpu().numpy().squeeze(0)
        preds = np.array(preds).transpose([1, 2, 0])
        preds = np.clip(preds, 0.0, 255.0).astype((np.uint8))

        output = pil_image.fromarray(preds)
        output.save(args.image_file.replace('.', args.model + '_x{}.'.format(args.scale)))                      


if __name__ == '__main__':
    main()


