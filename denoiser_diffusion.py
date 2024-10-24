import random
import argparse
import os
import torch
import numpy as np
from torch.optim import Adam, AdamW
import torch.optim.lr_scheduler as sched
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms, datasets
from cifar_dataset import faceset, DealDataset
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F
from copy import deepcopy
import cv2
from pathlib import Path
import torchvision
from pathlib import Path
from dataclasses import dataclass
from diffusers import UNet2DModel, DDPMPipeline, DDPMScheduler
from torch.utils.data.dataset import ConcatDataset
from diffusers.utils import make_image_grid
from accelerate import Accelerator, notebook_launcher
from sample import sample
from beta_schedule import linear_beta_schedule


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--exp', type=str, default='test')
    parser.add_argument('--model_path', type=str, default='checkpoint')
    parser.add_argument('--dataset_path', type=str, default='data')
    parser.add_argument('--ema', action='store_true')
    args = parser.parse_args()
    print(args)
    if not os.path.exists('./exp/' + args.exp):
        os.makedirs('./exp/' + args.exp)
    # writer = SummaryWriter('./exp/' + args.exp)
    model_path = args.model_path
    dataset_path_clean_1 = args.dataset_path

    print('model_path', model_path)
    print('dataset_path', dataset_path_clean_1)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())
    timesteps = 1000
    save_and_sample_every = 200
    seed = 0
    print('seed', seed)
    print('timesteps', timesteps)
    print('save_and_sample_every', save_and_sample_every)
    # setup_seed(seed)

    # define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps)
    noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps, variance_type='fixed_large', trained_betas=None)

    image_size, channels = 32, 3  # celeba-hq
    corruption_level = 0.5
    batch_size = 1500
    lr = 2e-4
    ema_decay = 0.999
    epochs = 5001
    sigma = 0.1
    min_loss = 10.0
    print("batch_size:", batch_size)
    print('lr', lr)
    print('ema_decay', ema_decay)


    @dataclass
    class TrainingConfig:
        image_size = image_size  # the generated image resolution
        train_batch_size = batch_size
        eval_batch_size = 9  # how many images to sample during evaluation
        num_epochs = epochs
        gradient_accumulation_steps = 1
        learning_rate = lr
        # lr_warmup_steps = 500
        save_image_epochs = 500
        save_model_epochs = 500
        mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
        output_dir = './exp/' + args.exp  # the model name locally and on the HF Hub
        overwrite_output_dir = True  # overwrite the old model when re-running the notebook
        seed = 0

    config = TrainingConfig()
    print('config', config)

    transform = Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
                # transforms.RandomHorizontalFlip(),
    ])


    total_dataset = DealDataset(folder=dataset_path_clean_1, transform=transform)

    cifar_clean_loader = torch.utils.data.DataLoader(dataset=total_dataset, batch_size=batch_size, shuffle=True)
    print('celeba shape', len(total_dataset), len(total_dataset))

    results_folder = Path('./exp/' + args.exp)
    results_folder.mkdir(exist_ok=True)

    model = UNet2DModel(
        sample_size=image_size,  # the target image resolution
        in_channels=channels,  # the number of input channels, 3 for RGB images
        out_channels=channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        norm_eps=1e-6,
        downsample_padding=0,
        flip_sin_to_cos=False,
        dropout=0.1,
        freq_shift=1,
        block_out_channels=(128, 256, 256, 256),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D"
        ),
    )
    model = torch.load(model_path, map_location='cpu')
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model parameters:', pytorch_total_params)
    optimizer = AdamW(model.parameters(), lr=lr)


    # args = (config, model, noise_scheduler, optimizer, celeba_train_loader, scheduler)
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
        if args.ema:
            model_ema = deepcopy(model).to(torch.device('cuda:0'))
            print('model_ema1', dict(model_ema.named_parameters()).keys(), model_ema.device)
            print('model.device', model.device)

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    print('model1', dict(model.named_parameters()).keys())
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, cifar_clean_loader)
    print('model2', dict(model.named_parameters()).keys())
    # accelerator.load_state(accelerator_path)
    global_step = 0
    print('model', model.device)
    if accelerator.is_main_process:
        print('Start samples...')
        samples_test = sample(model, None, timesteps, betas, image_size=image_size, batch_size=64, channels=channels)[-1]
        save_image((samples_test + 1) / 2, config.output_dir + '/start_sample.jpg', nrow=8, padding=0)

    # Now you train the model
    print('start training...', accelerator.device)
    for epoch in range(config.num_epochs):
        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            t = torch.randint(0, timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, t)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, t, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                # accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if args.ema and accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                ema_update(unwrapped_model, model_ema, ema_decay)

            logs = {"Epoch": epoch, "loss": loss.detach().item(), "step": global_step}
            accelerator.log(logs, step=global_step)
            global_step += 1
        if epoch % 20 == 0:
            print('logs:', logs)
        if epoch % save_and_sample_every == 0 and accelerator.is_main_process:
            print('Epoch samples...')
            samples_test = sample(model, None, timesteps, betas, image_size=image_size, batch_size=64, channels=channels)[-1]
            samples_test_ema = sample(model_ema, None, timesteps, betas, image_size=image_size, batch_size=64, channels=channels)[-1]
            print('min_loss, loss.item()', min_loss, loss.item())
            torch.save(model_ema, config.output_dir + '/epoch_' + str(epoch) + 'ema.pt')
            save_image((samples_test + 1) / 2, config.output_dir + '/' + str(epoch) + '-' + str(step) + 'sample.jpg', nrow=8, padding=0)
            save_image((samples_test_ema + 1) / 2, config.output_dir + '/' + str(epoch) + '-' + str(step) + 'sampleema.jpg',
                       nrow=8, padding=0)
        elif loss.item() < min_loss and accelerator.is_main_process:
            print('Best samples...')
            samples_test = sample(model, None, timesteps, betas, image_size=image_size, batch_size=64, channels=channels)[-1]
            samples_test_ema = sample(model_ema, None, timesteps, betas, image_size=image_size, batch_size=64, channels=channels)[-1]
            print('min_loss, loss.item()', min_loss, loss.item())
            min_loss = loss.item()
            save_image((samples_test + 1) / 2, config.output_dir + '/' + str(epoch) + '-' + str(step) + 'sample.jpg',
                       nrow=8, padding=0)
            save_image((samples_test_ema + 1) / 2, config.output_dir + '/' + str(epoch) + '-' + str(step) + 'sampleema.jpg',
                       nrow=8, padding=0)

    accelerator.end_training()