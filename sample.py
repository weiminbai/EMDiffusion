import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise



def p_losses(denoise_model, x_start, x_cond, matrix_inpainting1, matrix_inpainting2, t,
             sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l2", betas=None, sqrt_recip_alphas=None):
    distillation_loss, total_loss = 0., 0.
    alpha, temperature = 0.5, 1
    predicted_noise = 0.

    if noise is None:
        noise = torch.randn_like(x_start)

    betas_t = extract(betas, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x_start.shape)
    x_noisy = q_sample(x_start=x_start, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)

    predicted_noise = denoise_model(x_noisy, t).sample  # torch.Size([512, 1, 28, 28])
    predicted_signal_mean = sqrt_recip_alphas_t * (x_noisy - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    if loss_type == 'l1':
        pass
        # loss = F.l1_loss(noise, predicted_noise) + F.l1_loss(x_start, x_end)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        pass
        # loss = F.smooth_l1_loss(x_noisy - x_end, noise) + F.smooth_l1_loss(x_cond, x_cond_hat)
    else:
        raise NotImplementedError()


    return loss, predicted_signal_mean

@torch.no_grad()
def p_sample(model, x_cond, x, t, t_index, betas):
    # print('x, x_cond, t', x.shape, x_cond.shape, t.shape)
    betas_t = extract(betas, t, x.shape)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor, model(x, t, x_cond)) to predict the mean
    # TODO: Here, in the dwf, model predict the data (weimin)
    # x0_pred = model(x, t).sample
    # noise_pred = (x - (torch.sqrt(1 - sqrt_one_minus_alphas_cumprod_t ** 2) * x0_pred)) / sqrt_one_minus_alphas_cumprod_t
    noise_pred = model(x, t).sample
    # print(t.device, sqrt_recip_alphas_t.device, x.device, betas_t.device, noise_pred.device, sqrt_one_minus_alphas_cumprod_t.device)
    # cpu cuda:0 cpu cuda:0 cpu
    model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, x_cond, timesteps, betas, shape):

    b = shape[0]
    # print('b', b)
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=model.device)
    imgs = []

    # for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
    for i in reversed(range(0, timesteps)):
        img = p_sample(model, x_cond, img, torch.full((b,), i, dtype=torch.long, device=model.device), i, betas)
        imgs.append(img)
    return imgs


@torch.no_grad()
def sample(model, x_cond, timesteps, betas, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, x_cond, timesteps, betas, shape=(batch_size, channels, image_size, image_size))

