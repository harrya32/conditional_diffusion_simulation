import torch
import numpy as np
import functools


def marginal_prob_std(t, sigma_min, sigma_max):
    """Compute the standard deviation of $p_{0t}(x(t) | x(0))$.
    
    Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  

    Returns:
    The standard deviation.
    """    
    t = torch.tensor(t)
    std = sigma_min * (sigma_max / sigma_min) ** t
    #torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))
    return std

def diffusion_coeff(t, sigma_min, sigma_max):
    """Compute the diffusion coefficient of our SDE.

    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The vector of diffusion coefficients.
    """
    sigma = sigma_min * (sigma_max_2D / sigma_min) ** t
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max_2D) - np.log(sigma_min))))
    #torch.tensor(sigma**t)
    return diffusion
  
sigma_min = 0.01
sigma_max_2D = 8
marginal_prob_std_fn_2D = functools.partial(marginal_prob_std, sigma_min=sigma_min, sigma_max = sigma_max_2D)
diffusion_coeff_fn_2D = functools.partial(diffusion_coeff, sigma_min=sigma_min, sigma_max = sigma_max_2D)

sigma_max_BOD = 1
marginal_prob_std_fn_BOD = functools.partial(marginal_prob_std, sigma_min=sigma_min, sigma_max = sigma_max_BOD)
diffusion_coeff_fn_BOD = functools.partial(diffusion_coeff, sigma_min=sigma_min, sigma_max = sigma_max_BOD)