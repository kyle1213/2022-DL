import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import calc_patch_size, convert_rgb_to_y


#@calc_patch_size
def train(output_path, images_dir, with_aug, scale=2, patch_size=19):
    h5_file = h5py.File(output_path, 'a')

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(images_dir))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_images = []

        if with_aug:
            for s in [1.0, 0.9, 0.8, 0.7, 0.6]:
                for r in [0, 90, 180, 270]:
                    tmp = hr.resize((int(hr.width * s), int(hr.height * s)), resample=pil_image.BICUBIC)
                    tmp = tmp.rotate(r, expand=True)
                    hr_images.append(tmp)
        else:
            hr_images.append(hr)

        for hr in hr_images:
            hr_width = (hr.width // scale) * scale
            hr_height = (hr.height // scale) * scale
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            lr = hr.resize((hr.width // scale, hr_height // scale), resample=pil_image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            #hr = convert_rgb_to_y(hr)
            #lr = convert_rgb_to_y(lr)

            for i in range(0, lr.shape[0] - patch_size + 1, patch_size):
                for j in range(0, lr.shape[1] - patch_size + 1, patch_size):
                    lr_patches.append(lr[i:i+patch_size, j:j+patch_size])
                    hr_patches.append(hr[i*scale:i*scale+patch_size*scale, j*scale:j*scale+patch_size*scale])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    print(len(lr_patches))

    h5_file.close()


def eval(output_path, images_dir, scale=2):
    h5_file = h5py.File(output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // scale) * scale
        hr_height = (hr.height // scale) * scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // scale, hr_height // scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        #hr = convert_rgb_to_y(hr)
        #lr = convert_rgb_to_y(lr)
        print(i)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    train(output_path='D:/datasets/SR task/4xtrain_rgb_h5.h5', images_dir='D:/datasets/SR task/train/DIV2K/DIV2K_train_HR', with_aug=False, scale=4, patch_size=32)
    print("hi")
    #train(output_path='D:/datasets/SR task/train_h5.h5', images_dir='D:/datasets/SR task/train/Flickr2K', with_aug=False, scale=2, patch_size=32)
    print("hi")
    eval(output_path='D:/datasets/SR task/4xtest_rgb_h5.h5', images_dir='D:/datasets/SR task/test/DIV2K_valid', scale=4)
