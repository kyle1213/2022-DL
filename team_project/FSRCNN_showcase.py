import torch
import torch.backends.cudnn as cudnn

import numpy as np
import PIL.Image as pil_image
from PIL import Image

from models import FSRCNN
from utils import vid_convert_ycbcr_to_rgb, preprocess, calc_psnr

import time
import cv2
from multiprocessing import Process


def main(frames, ycbcrs):
    weights_file = "D:/User_Data/Desktop/github/2022-DL/team_project/models/SRCNN/x2/best.pth"

    model = FSRCNN(scale_factor=scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    with torch.no_grad():
        preds = model(frames).clamp(0.0, 1.0)
    preds = preds.mul(255.0).cpu().numpy().squeeze(1)

    output = np.array([preds, ycbcrs[..., 1].cpu().numpy(), ycbcrs[..., 2].cpu().numpy()]).transpose([1, 2, 3, 0])   # shape: 3, 30, 1440, 2560 - > ###

    output = np.clip(vid_convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)

    return output

if __name__ == '__main__':
    filepath = 'D:/User_Data/Desktop/github/2022-DL/team_project/test/vid4.mp4'
    video = cv2.VideoCapture(filepath) #'' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    scale = 2

    if not video.isOpened():
        print("Could not Open :", filepath)
        exit(0)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    count = 0

    frames = []

    while(video.isOpened()):
        ret, frame = video.read()

        frames.append(frame)
        hr_frames = []

        if len(frames) == 30:
            start = time.time()
            for i, f in enumerate(frames):
                frame = Image.fromarray(f)
                bicubic = np.array(frame.resize((frame.width * scale, frame.height * scale), resample=pil_image.BICUBIC))

                frame, _ = preprocess(frame, device)
                _, ycbcr = preprocess(bicubic, device)

                if i == 0:
                    inputs = frame
                    ycbcrs = torch.from_numpy(ycbcr).to(device).unsqueeze(0)
                    bicubic_frames = np.expand_dims(bicubic, 0)
                else:
                    inputs = torch.cat((inputs, frame))
                    ycbcrs = torch.cat((ycbcrs, torch.from_numpy(ycbcr).to(device).unsqueeze(0)))
                    bicubic_frames = np.append(bicubic_frames, np.expand_dims(bicubic, 0), axis=0)

            hr = main(inputs, ycbcrs.cpu())
            concat_img = np.append(bicubic_frames, hr, axis=2)
            print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

            for f in range(len(hr)):
                cv2.namedWindow('hr', flags=cv2.WINDOW_NORMAL)
                cv2.resizeWindow(winname='hr', width=852 * scale, height=480 * scale)
                cv2.namedWindow('bicubic', flags=cv2.WINDOW_NORMAL)
                cv2.resizeWindow(winname='bicubic', width=852 * scale, height=480 * scale)
                cv2.imshow('hr', hr[f])
                cv2.imshow('bicubic', bicubic_frames[f])
                key = cv2.waitKey(30)

                if key == ord('q'):
                    break

            frames = []
            bicubic_frames = []
        else:
            pass

    video.release()
