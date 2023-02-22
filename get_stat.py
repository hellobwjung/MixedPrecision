import os
import argparse
import time
import glob
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf

from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def get_psnr_ssim(path):
    psnr, ssim = 0.0, 0.0

    gts = glob.glob(os.path.join(path, 'org*.png'))
    gts.sort()

    outputs = glob.glob(os.path.join(path, '*.png'))
    outputs = list(set(outputs) - set(gts))
    outputs.sort()


    print(path)
    for idx, (g, o) in enumerate(zip(gts, outputs)):
        image_gt = io.imread(g)
        image_output = io.imread(o)

        p = peak_signal_noise_ratio(image_gt, image_output, data_range=255.)
        s = structural_similarity(image_gt, image_output, data_range=255., multichannel=True)

        psnr += p
        ssim += s


        if (idx % 99) == 0:
            print(idx, ' / 6000: psnr: %.3f, ssim: %.3f' %(psnr/(idx+1), ssim/(idx+1)))




    #     print(idx, "\n ", g, "\n ", o)
    # print(len(outputs))

    return psnr/6000, ssim/6000


def main():
    fp16_path = os.path.join('outputs',
                             'weights',
                             'MC_rmsc_16bit_2_same_noise',
                             '00753_MC_rmsc_16bit_2_same_noise_2.19823e+02.tflite')

    fp32_path = os.path.join('outputs',
                             'weights',
                             'MC_rmsc_16bit_2_same_noise',
                             '00753_MC_rmsc_16bit_2_same_noise_2.19823e+02.tflite')

    psnr16, ssim16 = get_psnr_ssim(fp16_path)
    psnr32, ssim32 = get_psnr_ssim(fp32_path)

    print('fp16: PSNR:%.3f, SSIM:%.3f' % (psnr16, ssim16))
    print('fp32: PSNR:%.3f, SSIM:%.3f' % (psnr32, ssim32))


    print('done done')


if __name__ == "__main__":
    main()
