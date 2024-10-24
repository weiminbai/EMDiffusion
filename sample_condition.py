from functools import partial
import os
import argparse
import yaml
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from lpips import LPIPS
import lpips
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from diffusers import UNet2DModel
from torchvision.utils import save_image
import numpy as np
import random

def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--shuffle', type=int)
    parser.add_argument('--use_gt_model', type=int)
    parser.add_argument('--gt_model_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--batch_num', type=int)
    parser.add_argument('--start', type=int)
    parser.add_argument('--use_ffhq', type=int, default=0)
    parser.add_argument('--ffhq_task_config', type=str)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    setup_seed(seed=args.seed)
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)
    print(device)
    
    # Load configurations
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    if args.use_ffhq==1:
        task_config = load_yaml(args.ffhq_task_config)
    # Load model
    print(args.use_gt_model)
    print(args)
    if args.use_gt_model:
        print(2222222)
        model = UNet2DModel.from_pretrained(args.gt_model_name)
        model = model.to(device)
        model.eval()

    else:
        print(1111111)
        model = torch.load(args.model_path, map_location=device)

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")
    print("retrieved????")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
    print("retrieved!!!!")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([
                                    Resize([64, 64], antialias=True),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    # transforms.Normalize((0.5), (0.5))
                                    ])
    print(args.start)
    print(type(args.start))
    # Exception) In case of inpainting, we need to generate a mask
    mask_gen = None
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
            **measure_config['mask_opt']
        )
    print('args.batch_size', args.batch_size)
    dataset = get_dataset(**data_config, transforms=transform, mask_gen=mask_gen)
    print('len(dataset)1', len(dataset))
    indices = list(range(args.start, args.start+2500))
    subset = Subset(dataset, indices)
    print('len(subset)', len(subset))
    loader = get_dataloader(subset, batch_size=args.batch_size, num_workers=0, train=bool(args.shuffle))

    
    
    # Do Inference
    cnt=0
    p0 = 0
    s0 = 0
    m0 = 0
    lp = 0
    loss_fn = LPIPS(net='alex', version='0.1')
    print('len(loader)', len(loader))
    for i, batch in enumerate(loader):
        # print('batch', len(batch))
        ref_img, random_mask, noisy_img = batch[0], batch[1], batch[2]
        # print('ref_img, random_mask', ref_img.shape, random_mask.shape)
        if i < args.batch_num:
            logger.info(f"Inference for image {i}")
            fname = str(i).zfill(5) + '.png'
            ref_img, random_mask, noisy_img = ref_img.to(device), random_mask.to(device), noisy_img.to(device)
            masks = random_mask
            print("retrieved?")
            # Exception) In case of inpainging,
            if measure_config['operator']['name'] == 'inpainting':
                print('random_mask', random_mask.shape, type(random_mask))
                measurement_cond_fn = partial(cond_method.conditioning, mask=masks)
                sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
                # Forward measurement model (Ax + n)
                y = operator.forward(ref_img, mask=masks)
                y_n = noiser(y, i)

            else: 
                # # Forward measurement model (Ax + n)
                y = operator.forward(ref_img)
                y_n = noiser(y, num=i)
                # y_n = noisy_img

            
            # Sampling
            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)
            for i in range(ref_img.shape[0]):
                plt.imsave(os.path.join(out_path, 'input', str(cnt)+ fname), clear_color(y_n[i]))
                plt.imsave(os.path.join(out_path, 'progress', str(cnt) + fname), clear_color(masks[i]))
                plt.imsave(os.path.join(out_path, 'label', str(cnt)+ fname), clear_color(ref_img[i]))
                plt.imsave(os.path.join(out_path, 'recon', str(cnt)+ fname), clear_color(sample[i]))
                cnt += 1


if __name__ == '__main__':
    main()
