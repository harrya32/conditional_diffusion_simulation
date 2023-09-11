import torch
import numpy as np
import functools


def marginal_prob_std(t, sigma_min, sigma_max):
  
    t = torch.tensor(t)
    std = sigma_min * (sigma_max / sigma_min) ** t
    return std

def diffusion_coeff(t, sigma_min, sigma_max):

    sigma = sigma_min * (sigma_max / sigma_min) ** t
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min))))
    return diffusion
  
sigma_min = 0.01
sigma_max_2D = 8
marginal_prob_std_fn_2D = functools.partial(marginal_prob_std, sigma_min=sigma_min, sigma_max = sigma_max_2D)
diffusion_coeff_fn_2D = functools.partial(diffusion_coeff, sigma_min=sigma_min, sigma_max = sigma_max_2D)

sigma_max_BOD = 13
marginal_prob_std_fn_BOD = functools.partial(marginal_prob_std, sigma_min=sigma_min, sigma_max = sigma_max_BOD)
diffusion_coeff_fn_BOD = functools.partial(diffusion_coeff, sigma_min=sigma_min, sigma_max = sigma_max_BOD)

sigma_max_MNIST = 25
marginal_prob_std_fn_MNIST = functools.partial(marginal_prob_std, sigma_min=sigma_min, sigma_max = sigma_max_MNIST)
diffusion_coeff_fn_MNIST = functools.partial(diffusion_coeff, sigma_min=sigma_min, sigma_max = sigma_max_MNIST)