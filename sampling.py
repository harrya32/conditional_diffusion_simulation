from scipy import integrate
import torch
import numpy as np
from tqdm import notebook
from torch.distributions import MultivariateNormal
from torch.distributions import Normal
import math

def Euler_Maruyama_sampler_2D(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=1000, 
                           num_steps=1000, 
                           device='cpu', 
                           eps=1e-3):

    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 2, device=device) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            batch_time_step = torch.reshape(batch_time_step, (x.shape[0], 1))
            x_with_t = torch.hstack([x, batch_time_step])
            mean_x = x + (g**2)[:, None] * score_model(x_with_t) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x)      

    return mean_x



signal_to_noise_ratio = 0.16 

def pc_sampler_2D(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=2048, 
               num_steps=1000, 
               snr=signal_to_noise_ratio,                
               device='cpu',
               eps=1e-3):

    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 2, device=device) * marginal_prob_std(t)[:, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            batch_time_step_ = torch.reshape(batch_time_step, (x.shape[0], 1))
            x_with_t = torch.hstack([x, batch_time_step_])
            # Corrector step (Langevin MCMC)
            grad = score_model(x_with_t)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)     
            x_with_t = torch.hstack([x, batch_time_step_])

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None] * score_model(x_with_t) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None] * torch.randn_like(x)      

    # The last step does not include any noise
    return x_mean

#@title Define the ODE sampler (double click to expand or collapse)



## The error tolerance for the black-box ODE solver
error_tolerance = 1e-6 #@param {'type': 'number'}
def ode_sampler_2D(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=2048, 
                atol=error_tolerance, 
                rtol=error_tolerance, 
                device='cpu', 
                z=None,
                eps=1e-3):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
    """
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
        init_x = torch.randn(batch_size, 2, device=device) * marginal_prob_std(t)[:, None]
    else:
        init_x = z

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))  
        time_steps = torch.reshape(time_steps, (sample.shape[0], 1))
        sample = torch.hstack([sample, time_steps])

        with torch.no_grad():    
            score = score_model(sample)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):        
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t    
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x

def CDE_Euler_Maruyama_sampler_2D(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           y_obs,
                           batch_size=10000, 
                           num_steps=1000, 
                           eps=1e-3):

    t = torch.ones(batch_size)
    init_x = torch.randn(batch_size, 1) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    y_obs = y_obs.repeat(batch_size)
    y_obs = y_obs.reshape(batch_size,1)
    x = torch.hstack([init_x, y_obs])
    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps): 

            batch_time_step = torch.ones(batch_size) * time_step
            g = diffusion_coeff(batch_time_step)
            batch_time_step_ = torch.reshape(batch_time_step, (x.shape[0], 1))
            x_with_t = torch.hstack([x, batch_time_step_])
            mean_x = x + (g**2)[:, None] * score_model(x_with_t) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x) 
            x = torch.hstack([x,y_obs])
            x = x[:, [0,2]]
            
    # Do not include any noise in the last sampling step.
    return mean_x

def CDE_pc_sampler_2D(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               y_obs,
               batch_size=2048, 
               num_steps=1000, 
               snr=signal_to_noise_ratio,               
               eps=1e-3):

    t = torch.ones(batch_size)
    init_x = torch.randn(batch_size, 1) * marginal_prob_std(t)[:, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    y_obs = y_obs.repeat(batch_size)
    y_obs = y_obs.reshape(batch_size,1)
    x = torch.hstack([init_x, y_obs])
    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size) * time_step
            batch_time_step_ = torch.reshape(batch_time_step, (x.shape[0], 1))
            x_with_t = torch.hstack([x, batch_time_step_])
            # Corrector step (Langevin MCMC)
            grad = score_model(x_with_t)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      
            
            x = torch.hstack([x,y_obs])
            x = x[:, [0,2]]
            x_with_t = torch.hstack([x, batch_time_step_])
            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None] * score_model(x_with_t) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None] * torch.randn_like(x)
            
            #conditional info
            x = torch.hstack([x,y_obs])
            x = x[:, [0,2]]

    # The last step does not include any noise
    return x_mean

def get_diffused_y_2D(y_obs, timesteps, marginal_prob_std):
    timesteps = torch.flip(timesteps, [0])
    sd = marginal_prob_std(timesteps).reshape(timesteps.shape[0], 1)
    diffused_y = y_obs + torch.randn(timesteps.shape[0], 1) * sd
    return diffused_y

def get_diffused_2D_old(data, n, diffusion_coeff):
    data = data.item()
    t = 0
    dt = 1/n
    diffused = [data]
    for i in range(n):
        g = diffusion_coeff(t).item()
        data += g * np.random.randn(1)[0] * np.sqrt(dt)
        t += dt
        diffused.append(data)
        
    return torch.tensor(diffused, dtype = torch.float32)[:,None]

sigma_min=0.01
sigma_max_2D=8
sigma_max_BOD=13

def sde_VE(x, t, sigma_min, sigma_max):
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    diffusion = sigma * np.sqrt(2 * (np.log(sigma_max) - np.log(sigma_min)))
    drift = 0
    return drift, diffusion

def get_diffused_2D(data, n, sde, sigma_min, sigma_max):
    data = data.clone().detach()
    data = data.item()
    t = 1e-5
    dt = 1/n
    diffused = [data]
    for i in range(n):
        drift, diffusion = sde(data, t, sigma_min, sigma_max)
        
        data += drift * dt
        data += diffusion * np.random.randn(1)[0] * np.sqrt(dt)
        
        t += dt
        
        diffused.append(data.copy())
    
    return torch.tensor(diffused, dtype = torch.float32)[:,None]

def CDiffE_Euler_Maruyama_sampler_2D(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           y_obs,
                           batch_size=10000, 
                           num_steps=1000,
                           eps=1e-3,
                           sigma_min = sigma_min,
                           sigma_max = sigma_max_2D):

    t = torch.ones(batch_size)
    init_x = torch.randn(batch_size, 2) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    #diff = get_diffused_2D(y_obs, num_steps, diffusion_coeff)
    #diffused_y = [i.repeat(batch_size).reshape(batch_size,1) for i in diff]
    diffused_y = [i.repeat(batch_size).reshape(batch_size,1) for i in \
                  get_diffused_2D(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):
            
            batch_time_step = torch.ones(batch_size) * time_step
            idx = num_steps - idx - 1
            y_obs_t = diffused_y[idx]
            x = torch.hstack([x,y_obs_t])[:, [0,2]]
            g = diffusion_coeff(batch_time_step)
            batch_time_step_ = torch.reshape(batch_time_step, (x.shape[0], 1))
            x_with_t = torch.hstack([x, batch_time_step_])
            mean_x = x + (g**2)[:, None] * score_model(x_with_t) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x) 

    return mean_x



def CDiffE_pc_sampler_2D(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               y_obs,
               batch_size=2048, 
               num_steps=1000, 
               snr=signal_to_noise_ratio,                
               eps=1e-3,
               sigma_min = sigma_min,
               sigma_max = sigma_max_2D):

    t = torch.ones(batch_size)
    init_x = torch.randn(batch_size, 2) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    #diff = get_diffused_2D(y_obs, num_steps, diffusion_coeff)
   
    #diffused_y = [i.repeat(batch_size).reshape(batch_size,1) for i in diff]
    diffused_y = [i.repeat(batch_size).reshape(batch_size,1) for i in \
                  get_diffused_2D(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    
    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):
            idx = num_steps - idx - 1
            y_obs_t = diffused_y[idx]
            batch_time_step = torch.ones(batch_size) * time_step
            batch_time_step_ = torch.reshape(batch_time_step, (x.shape[0], 1))
            x = torch.hstack([x,y_obs_t])[:, [0,2]]
            
            x_with_t = torch.hstack([x, batch_time_step_])
            # Corrector step (Langevin MCMC)
            grad = score_model(x_with_t)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      
            x_with_t = torch.hstack([x, batch_time_step_])

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None] * score_model(x_with_t) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None] * torch.randn_like(x)
            
    # The last step does not include any noise
    return x_mean

'''def SMCDiff_Euler_Maruyama_sampler_2D(score_model, marginal_prob_std, diffusion_coeff, y_obs, k, 
                                   num_steps=1000, eps=1e-3):
    
    t = torch.ones(k)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    weights = np.ones(k)
    x = torch.randn(k, 2) * marginal_prob_std(t)[:, None]
    diffused_y = [i.repeat(k).reshape(k,1) for i in get_diffused_y_2D(y_obs, time_steps, marginal_prob_std)]
    
    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):            
            idx = num_steps - idx - 1
            y_obs_t = diffused_y[idx]
            x = torch.hstack([x,y_obs_t])[:, [0,2]]
            
            if (idx - 1) >= 0:
                y_update_mean, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
                y_update_mean = y_update_mean[:,1]
                y_update_actual = diffused_y[idx-1].flatten()

                log_w = log_normal_density_2D(y_update_actual, y_update_mean, sd.flatten())
                log_w -= torch.logsumexp(log_w, 0)
                weights = torch.exp(log_w).cpu().detach().numpy()
                weights /= sum(weights)
                
                resample_index = systematic(weights, k)
                x = x[resample_index]
                weights = np.ones_like(weights)
            
            mu, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
            x = mu + sd * torch.randn_like(x) 
            
    return mu'''

def SMCDiff_pc_sampler_2D(score_model, marginal_prob_std, diffusion_coeff, y_obs, k, snr=signal_to_noise_ratio, 
                          num_steps=1000, eps=1e-3, sigma_min = sigma_min, sigma_max = sigma_max_2D):

    t = torch.ones(k)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    weights = np.ones(k)/k
    xs = []
    init_x = torch.randn(k, 2) * marginal_prob_std(t)[:, None]
    xs.append(init_x)
    #diffused_y = [i.repeat(k).reshape(k,1) for i in get_diffused_y_2D(y_obs, time_steps, marginal_prob_std)]
    #diff = get_diffused_2D(y_obs, num_steps, diffusion_coeff)
    #diffused_y = [i.repeat(k).reshape(k,1) for i in diff]
    diffused_y = [i.repeat(k).reshape(k,1) for i in \
                  get_diffused_2D(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    
    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):
            idx = num_steps - idx - 1
            batch_time_step = torch.ones(k) * time_step
            y_obs_t = diffused_y[idx]
            x = xs[-1]
            x = torch.hstack([x,y_obs_t])[:, [0,2]]
                        
            if (idx - 1) >= 0:
                y_update_mean, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
                y_update_mean = y_update_mean[:,1]
                y_update_actual = diffused_y[idx-1].flatten()


                log_w = log_normal_density_2D(y_update_actual, y_update_mean, sd.flatten())
                log_w -= torch.logsumexp(log_w, 0)

                weights = torch.exp(log_w).cpu().detach().numpy()
                weights /= sum(weights) 
                
                resample_index = systematic(weights, k)
                x = x[resample_index]
                weights = np.ones_like(weights)
                
            #corrector step (Langevin MCMC)
            batch_time_step = torch.reshape(torch.ones(k) * time_step, (x.shape[0], 1))
            x_with_t = torch.hstack([x, batch_time_step])
            grad = score_model(x_with_t)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
            x = torch.hstack([x,y_obs_t])[:, [0,2]]
            
            mu, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
            x_t_1 = mu + sd * torch.randn_like(x) 
            xs.append(x_t_1)

    return xs[-1]

def SMCDiff_Euler_Maruyama_sampler_2D(score_model, marginal_prob_std, diffusion_coeff, y_obs, k, snr=signal_to_noise_ratio,
                                      num_steps=1000, eps=1e-3, sigma_min=sigma_min, sigma_max=sigma_max_2D):

    t = torch.ones(k)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    weights = np.ones(k)/k
    xs = []
    init_x = torch.randn(k, 2) * marginal_prob_std(t)[:, None]
    xs.append(init_x)
    #diff = get_diffused_2D(y_obs, num_steps, diffusion_coeff)
    #diffused_y = [i.repeat(k).reshape(k,1) for i in diff]
    diffused_y = [i.repeat(k).reshape(k,1) for i in \
                  get_diffused_2D(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):
            idx = num_steps - idx - 1
            batch_time_step = torch.ones(k) * time_step
            y_obs_t = diffused_y[idx]
            x = xs[-1]
            x = torch.hstack([x,y_obs_t])[:, [0,2]]
                        
            if (idx - 1) >= 0:
                y_update_mean, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
                y_update_mean = y_update_mean[:,1]
                y_update_actual = diffused_y[idx-1].flatten()


                log_w = log_normal_density_2D(y_update_actual, y_update_mean, sd.flatten())
                log_w -= torch.logsumexp(log_w, 0)

                weights = torch.exp(log_w).cpu().detach().numpy()
                weights /= sum(weights) 
                
                resample_index = systematic(weights, k)
                x = x[resample_index]
                weights = np.ones_like(weights)
                
            mu, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
            x_t_1 = mu + sd * torch.randn_like(x) 
            xs.append(x_t_1)

    return xs[-1]

def log_normal_density_2D(sample, mean, sd):
    return Normal(loc=mean, scale=sd).log_prob(sample)
#################
#################
#################

signal_to_noise_ratio = 0.16 

def pc_sampler_BOD(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=10000, 
               num_steps=1000, 
               snr=signal_to_noise_ratio,
               eps=1e-5):

    t = torch.ones(batch_size)
    init_x = torch.randn(batch_size, 7) * marginal_prob_std(t)[:, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    
    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size) * time_step
            batch_time_step_ = torch.reshape(batch_time_step, (x.shape[0], 1))
            x_with_t = torch.hstack([x, batch_time_step_])
            
            # Corrector step (Langevin MCMC)
            grad = score_model(x_with_t)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x) 

            # Predictor step (Euler-Maruyama)
            mu, sd = get_next_x(x, batch_size, score_model, diffusion_coeff, time_step, torch.tensor(step_size))
            x = mu + sd * torch.randn_like(x) 

    return mu


def Euler_Maruyama_sampler_BOD(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=10000, 
                           num_steps=1000, 
                           eps=1e-5):

    t = torch.ones(batch_size)
    init_x = torch.randn(batch_size, 7) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps):

            mu, sd = get_next_x(x, batch_size, score_model, diffusion_coeff, time_step, step_size)
            x = mu + sd * torch.randn_like(x) 


    return mu

def get_next_x(x, batch_size, score_model, diffusion_coeff, time_step, step_size):

    batch_time_step = torch.ones(batch_size) * time_step
    g = diffusion_coeff(batch_time_step)
    batch_time_step_ = torch.reshape(batch_time_step, (x.shape[0], 1))
    x_with_t = torch.hstack([x, batch_time_step_])
    
    score = score_model(x_with_t)
    mean_x = x + (g**2)[:, None] * score * step_size
    sd = torch.sqrt(step_size) * g[:, None]
    
    return mean_x, sd


def get_diffused_y(y_obs, timesteps, marginal_prob_std):
    timesteps = torch.flip(timesteps, [0])
    sd = marginal_prob_std(timesteps).reshape(timesteps.shape[0], 1)
    diffused_y = y_obs + torch.randn(timesteps.shape[0], y_obs.shape[0]) * sd
    return diffused_y

def get_diffused_BOD_old(obs, n, diffusion_coeff):
    data = obs.clone().detach()
    t = 0
    dt = 1/n
    diffused = [data.clone().detach()]
    for i in range(n):
        g = diffusion_coeff(t).reshape(1)
        data += g * torch.randn(5) * np.sqrt(dt)
        t += dt
        diffused.append(data.clone().detach())
        
    return torch.vstack(diffused)

def get_diffused_BOD(obs, n, sde, sigma_min, sigma_max):
    data = obs.clone().detach()
    t = 1e-5
    dt = 1/n
    diffused = [data.clone().detach()]
    for i in range(n):
        drift, diffusion = sde(data, t, sigma_min, sigma_max)
        data += drift * dt
        data += diffusion * torch.randn(5) * np.sqrt(dt)
        diffused.append(data.clone().detach())
        t += dt

    return torch.vstack(diffused)


def CDiffE_Euler_Maruyama_sampler_BOD(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           y_obs,
                           batch_size=10000, 
                           num_steps=1000, 
                           eps=1e-5, sigma_min=sigma_min, sigma_max=sigma_max_BOD):

    t = torch.ones(batch_size)
    x = torch.randn(batch_size, 7) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    #diff = get_diffused_BOD(y_obs, num_steps, diffusion_coeff)
    #diffused_y = [i.repeat(batch_size).reshape(batch_size,5) for i in diff]
    diffused_y = [i.repeat(batch_size).reshape(batch_size,5) for i in \
                  get_diffused_BOD(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]

    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):
            idx = num_steps - idx - 1
            y_obs_t = diffused_y[idx]
            x = torch.hstack([x,y_obs_t])[:, [0,1,7,8,9,10,11]]
            mu, sd = get_next_x(x, batch_size, score_model, diffusion_coeff, time_step, step_size)
            x = mu + sd * torch.randn_like(x) 

    return mu

signal_to_noise_ratio = 0.16

def CDiffE_pc_sampler_BOD(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               y_obs,
               batch_size=2048, 
               num_steps=1000, 
               snr=signal_to_noise_ratio,
               eps=1e-5, sigma_min=sigma_min, sigma_max=sigma_max_BOD):

    t = torch.ones(batch_size)
    init_x = torch.randn(batch_size, 7) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    diffused_y = [i.repeat(batch_size).reshape(batch_size,5) for i in \
                  get_diffused_BOD(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    with torch.no_grad(): 
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):
            idx = num_steps - idx - 1
            y_obs_t = diffused_y[idx]
            x = torch.hstack([x,y_obs_t])[:, [0,1,7,8,9,10,11]]
            
            # Corrector step (Langevin MCMC)
            batch_time_step = torch.reshape(torch.ones(batch_size) * time_step, (x.shape[0], 1))
            x_with_t = torch.hstack([x, batch_time_step])
            grad = score_model(x_with_t)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      
            x = torch.hstack([x,y_obs_t])[:, [0,1,7,8,9,10,11]]
            
            # Predictor step (Euler-Maruyama)
            mu, sd = get_next_x(x, batch_size, score_model, diffusion_coeff, time_step, step_size)
            x = mu + sd * torch.randn_like(x) 
            
    return mu

def CDE_Euler_Maruyama_sampler_BOD(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               y_obs,
               batch_size=2048, 
               num_steps=1000, 
               eps=1e-5):

    t = torch.ones(batch_size)
    init_x = torch.randn(batch_size, 2) * marginal_prob_std(t)[:, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    y_obs = y_obs.repeat(batch_size)
    y_obs = y_obs.reshape(batch_size,5)
    x = torch.hstack([init_x, y_obs])

    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps):
            
            # Predictor step
            x = cde_get_next_x(x, y_obs, batch_size, score_model, diffusion_coeff, time_step, torch.tensor(step_size))
            
    return x

def CDE_pc_sampler_BOD(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               y_obs,
               batch_size=2048, 
               num_steps=1000, 
               snr=signal_to_noise_ratio,
               eps=1e-5):

    t = torch.ones(batch_size)
    init_x = torch.randn(batch_size, 2) * marginal_prob_std(t)[:, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    y_obs = y_obs.repeat(batch_size)
    y_obs = y_obs.reshape(batch_size,5)
    x = torch.hstack([init_x, y_obs])

    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size) * time_step
            batch_time_step_ = torch.reshape(batch_time_step, (x.shape[0], 1))
            x_with_t = torch.hstack([x, batch_time_step_])
            # Corrector step (Langevin MCMC)
            grad = score_model(x_with_t)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x[:,[0,1]]
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)    
            x = torch.hstack([x, y_obs])
            
            # Predictor step
            x = cde_get_next_x(x, y_obs, batch_size, score_model, diffusion_coeff, time_step, torch.tensor(step_size))
            
    return x

def cde_get_next_x(x, y_obs, batch_size, score_model, diffusion_coeff, time_step, step_size):

    batch_time_step = torch.ones(batch_size) * time_step
    g = diffusion_coeff(batch_time_step)
    batch_time_step_ = torch.reshape(batch_time_step, (x.shape[0], 1))
    x_with_t = torch.hstack([x, batch_time_step_])
    score = score_model(x_with_t)
    
    x = x[:,[0,1]]
    
    mean_x = x + (g**2)[:, None] * score * step_size
    sd = torch.sqrt(step_size) * g[:, None]
    x = mean_x + sd * torch.randn_like(x)
    x = torch.hstack([x, y_obs])
    return x


def log_normal_density(sample, mean, sd):
    sample = sample[0]
    sd = sd[0]
    cov = sd * torch.eye(sample.shape[0])
    return MultivariateNormal(loc=mean, covariance_matrix=cov).log_prob(sample)

def SMCDiff_Euler_Maruyama_sampler_BOD(score_model, marginal_prob_std, diffusion_coeff, y_obs, k, num_steps=1000, 
                                       eps=1e-5, snr = signal_to_noise_ratio, sigma_min=sigma_min, sigma_max=sigma_max_BOD):

    t = torch.ones(k)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    weights = np.ones(k)
    xs = []
    init_x = torch.randn(k, 7) * marginal_prob_std(t)[:, None]
    xs.append(init_x)
    #diff = get_diffused_BOD(y_obs, num_steps, diffusion_coeff)
    #diffused_y = [i.repeat(k).reshape(k,5) for i in diff]
    diffused_y = [i.repeat(k).reshape(k,5) for i in get_diffused_BOD(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    
    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):
            idx = num_steps - idx - 1
            batch_time_step = torch.ones(k) * time_step
            y_obs_t = diffused_y[idx]
            x = xs[-1]
            x = torch.hstack([x,y_obs_t])[:, [0,1,7,8,9,10,11]]
            
            
            #get predicted y_{t-1}
            if (idx - 1) >= 0:
                y_update_mean, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
                y_update_mean = y_update_mean[:,[2,3,4,5,6]]
                y_update_actual = diffused_y[idx-1]

                # compute un-normalized weighting factor for importance resampling step
                log_w = log_normal_density(y_update_actual, y_update_mean, sd)
                log_w -= torch.logsumexp(log_w, 0)

                # Update Self-normalized importance weights
                weights = torch.exp(log_w).cpu().detach().numpy()
                weights /= sum(weights) # Re-normalize
                
                resample_index = systematic(weights, k)
                x = x[resample_index]
                weights = np.ones_like(weights)
            
            mu, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
            x_t_1 = mu + sd * torch.randn_like(x) 
            xs.append(x_t_1)

    return xs[-1]

def SMCDiff_pc_sampler_BOD(score_model, marginal_prob_std, diffusion_coeff, y_obs, k, num_steps=1000, 
                           eps=1e-5, snr = signal_to_noise_ratio, sigma_min=sigma_min, sigma_max=sigma_max_BOD):

    t = torch.ones(k)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    weights = np.ones(k)
    xs = []
    init_x = torch.randn(k, 7) * marginal_prob_std(t)[:, None]
    xs.append(init_x)
    diffused_y = [i.repeat(k).reshape(k,5) for i in get_diffused_BOD(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    
    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):
            idx = num_steps - idx - 1
            batch_time_step = torch.ones(k) * time_step
            y_obs_t = diffused_y[idx]
            x = xs[-1]
            x = torch.hstack([x,y_obs_t])[:, [0,1,7,8,9,10,11]]
            
            #corrector step (Langevin MCMC)
            batch_time_step = torch.reshape(torch.ones(k) * time_step, (x.shape[0], 1))
            x_with_t = torch.hstack([x, batch_time_step])
            grad = score_model(x_with_t)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
            
            x = torch.hstack([x,y_obs_t])[:, [0,1,7,8,9,10,11]]
            
            #get predicted y_{t-1}
            if (idx - 1) >= 0:
                y_update_mean, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
                y_update_mean = y_update_mean[:,[2,3,4,5,6]]
                y_update_actual = diffused_y[idx-1]

                # compute un-normalized weighting factor for importance resampling step
                log_w = log_normal_density(y_update_actual, y_update_mean, sd)
                log_w -= torch.logsumexp(log_w, 0)

                # Update Self-normalized importance weights
                weights = torch.exp(log_w).cpu().detach().numpy()
                weights /= sum(weights) # Re-normalize
                
                resample_index = systematic(weights, k)
                x = x[resample_index]
                weights = np.ones_like(weights)
            
            mu, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
            x_t_1 = mu + sd * torch.randn_like(x) 
            xs.append(x_t_1)

    return xs[-1]

def residual(W):
    """Residual resampling.
    """
    N = W.shape[0]
    M = N
    A = np.empty(M, dtype=np.int64)
    MW = M * W
    intpart = np.floor(MW).astype(np.int64)
    sip = np.sum(intpart)
    res = MW - intpart
    sres = M - sip
    A[:sip] = np.arange(N).repeat(intpart)
    # each particle n is repeated intpart[n] times
    if sres > 0:
        A[sip:] = multinomial(res / sres, M=sres)
    return A

def systematic(W, M):
    """Systematic resampling.
    """
    su = (np.random.rand(1) + np.arange(M)) / M
    return inverse_cdf(su, W)

def multinomial(W, M):
    """Multinomial resampling.

    Popular resampling scheme, which amounts to sample N independently from
    the multinomial distribution that generates n with probability W^n. 

    This resampling scheme is *not* recommended for various reasons; basically
    schemes like stratified / systematic / SSP tends to introduce less noise,
    and may be faster too (in particular systematic). 
    """
    return inverse_cdf(uniform_spacings(M), W)

def uniform_spacings(N):
    """Generate ordered uniform variates in O(N) time.

    Parameters
    ----------
    N: int (>0)
        the expected number of uniform variates

    Returns
    -------
    (N,) float ndarray
        the N ordered variates (ascending order)

    Note
    ----
    This is equivalent to::

        from numpy import random
        u = sort(random.rand(N))

    but the line above has complexity O(N*log(N)), whereas the algorithm
    used here has complexity O(N).

    """
    z = np.cumsum(-np.log(np.random.rand(N + 1)))
    return z[:-1] / z[-1]

def inverse_cdf(su, W):
    """Inverse CDF algorithm for a finite distribution.

        Parameters
        ----------
        su: (M,) ndarray
            M sorted uniform variates (i.e. M ordered points in [0,1]).
        W: (N,) ndarray
            a vector of N normalized weights (>=0 and sum to one)

        Returns
        -------
        A: (M,) ndarray
            a vector of M indices in range 0, ..., N-1
    """
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, dtype=np.int64)
    for n in range(M):
        while su[n] > s:
            if j < W.shape[0]-1:
                j += 1
            s += W[j]
        A[n] = j
    return A