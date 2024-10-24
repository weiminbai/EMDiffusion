from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import os
import argparse
from lpips import LPIPS
import lpips
import json


parser = argparse.ArgumentParser()
parser.add_argument('--p1', type=str)
parser.add_argument('--p2', type=str)
args = parser.parse_args()

## Initializing the model
loss_fn = LPIPS(net='alex',version='0.1')
path1 = args.p1 + '/'
path2 = args.p2 + '/'
files_in_folder = os.listdir(path1)
p0=0
s0=0
m0=0
lp=0
num = len(files_in_folder)
print(num)
for i in range(num):

    img1 = cv2.imread(path1+files_in_folder[i])
    img2 = cv2.imread(path2+files_in_folder[i])
    # Load images
    lpips_img1 = lpips.im2tensor(lpips.load_image(path1+files_in_folder[i]))  # RGB image from [-1,1]
    lpips_img2 = lpips.im2tensor(lpips.load_image(path2+files_in_folder[i]))
    dist01 = loss_fn.forward(lpips_img1, lpips_img2)
    
    p = compare_psnr(img1, img2)
    # print(img1.shape)
    s = compare_ssim(img1, img2, channel_axis=-1)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
    m = compare_mse(img1, img2)
    p0+=p
    s0+=s
    m0+=m
    lp+=dist01
    print(files_in_folder[i], p)

    # print('img{}: PSNR:{},SSIM:{},MSE:{}'.format(i, p, s, m))
# print(path1)

print('Final PSNR:{},SSIM:{},MSE:{},LPIPS:{}'.format(p0/num, s0/num, m0/num, lp/num))
