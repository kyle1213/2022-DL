import torch
import torch.backends.cudnn as cudnn

import numpy as np
import PIL.Image as pil_image
from PIL import Image

from models import groupedFSRCNN
from utils import vid_convert_ycbcr_to_rgb, rgbpreprocess, calc_psnr

import time
import cv2
from multiprocessing import Process


def main(frames):
    weights_file = "D:/User_Data/Desktop/github/2022-DL/team_project/models/groupedSRCNN/x2/best.pth"

    model = groupedFSRCNN(scale_factor=scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    with torch.no_grad():
        preds = model(frames).clamp(0.0, 1.0)

    preds = preds.mul(255.0).cpu().numpy()
    preds = np.array(preds).transpose([0, 2, 3, 1])
    preds = np.clip(preds, 0.0, 255.0).astype((np.uint8))

    return preds


if __name__ == '__main__':
    filepath = 'D:/User_Data/Desktop/github/2022-DL/team_project/test/vid4.mp4'
    video = cv2.VideoCapture(filepath)  # '' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함

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

    while (video.isOpened()):
        ret, frame = video.read()

        frames.append(frame)
        hr_frames = []
        if not ret:  # 없으면
            if len(frames) > 1:
                frames = frames[:len(frames)-1]
                for i, f in enumerate(frames):
                    frame = Image.fromarray(f)
                    bicubic = np.array(frame.resize((frame.width * scale, frame.height * scale), resample=pil_image.BICUBIC))
                    frame = rgbpreprocess(frame, device)

                    if i == 0:
                        inputs = frame
                        bicubic_frames = np.expand_dims(bicubic, 0)
                    else:
                        inputs = torch.cat((inputs, frame))
                        bicubic_frames = np.append(bicubic_frames, np.expand_dims(bicubic, 0), axis=0)

                hr = main(inputs)

                for f in range(len(hr)):
                    cv2.namedWindow('hr', flags=cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(winname='hr', width=852*scale, height=480*scale)
                    cv2.namedWindow('bicubic', flags=cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(winname='bicubic', width=852*scale, height=480*scale)
                    cv2.imshow('hr', hr[f])
                    cv2.imshow('bicubic', bicubic_frames[f])
                    key = cv2.waitKey(30)

                    if key == ord('q'):
                        break

                frames = []
                bicubic_frames = []

                break
            else:
                break

        else:
            if len(frames) == 30:
                start = time.time()
                for i, f in enumerate(frames):
                    frame = Image.fromarray(f)
                    bicubic = np.array(frame.resize((frame.width * scale, frame.height * scale), resample=pil_image.BICUBIC))
                    frame = rgbpreprocess(frame, device)

                    if i == 0:
                        inputs = frame
                        bicubic_frames = np.expand_dims(bicubic, 0)
                    else:
                        inputs = torch.cat((inputs, frame))
                        bicubic_frames = np.append(bicubic_frames, np.expand_dims(bicubic, 0), axis=0)
                hr = main(inputs)

                print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

                for f in range(len(hr)):
                    cv2.namedWindow('hr', flags=cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(winname='hr', width=852*scale, height=480*scale)
                    cv2.namedWindow('bicubic', flags=cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(winname='bicubic', width=852*scale, height=480*scale)
                    cv2.imshow('hr', hr[f])
                    cv2.imshow('bicubic', bicubic_frames[f])
                    key = cv2.waitKey(30)

                    if key == ord('q'):
                        break

                frames = []
                bicubic_frames = []
            else:
                pass
        torch.cuda.empty_cache()
    video.release()
