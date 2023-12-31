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

    return x_mean


def CDE_Euler_Maruyama_sampler_2D(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           y_obs,
                           batch_size=10000, 
                           num_steps=1000, 
                           eps=1e-3):

    t = torch.ones(batch_size)
    x = torch.randn(batch_size, 1) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    y_obs = y_obs.repeat(batch_size)
    y_obs = y_obs.reshape(batch_size,1)

    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps): 

            batch_time_step = torch.ones(batch_size) * time_step
            g = diffusion_coeff(batch_time_step)[:, None]
            score = score_model(x, y_obs, batch_time_step[:,None])

            mean_x = x + g**2 * score * step_size
            x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x) 
            
    return mean_x

def CDE_pc_sampler_2D(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           y_obs,
                           batch_size=10000, 
                           num_steps=1000, 
                           eps=1e-3, snr=0.16):

    t = torch.ones(batch_size)
    x = torch.randn(batch_size, 1) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    y_obs = y_obs.repeat(batch_size)
    y_obs = y_obs.reshape(batch_size,1)

    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps): 

            batch_time_step = torch.ones(batch_size) * time_step
            g = diffusion_coeff(batch_time_step)[:, None]
            
            grad = score_model(x, y_obs, batch_time_step[:,None])
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)    
            
            score = score_model(x, y_obs, batch_time_step[:,None])
            mean_x = x + g**2 * score * step_size
            x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x) 
            
    return mean_x

sigma_min=0.01
sigma_max_2D=8
sigma_max_BOD=13
sigma_max_MNIST=25

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
                           sigma_max = sigma_max_2D,
                           diffused_y = None):

    t = torch.ones(batch_size)
    init_x = torch.randn(batch_size, 2) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    if diffused_y is None:
        diffused_y = [i.repeat(batch_size).reshape(batch_size,1) for i in \
                      get_diffused_2D(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    else:
        diffused_y = [i.repeat(batch_size).reshape(batch_size,1) for i in diffused_y]
        
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


def SMCDiff_pc_sampler_2D(score_model, marginal_prob_std, diffusion_coeff, y_obs, k, snr=signal_to_noise_ratio, 
                          num_steps=1000, eps=1e-3, sigma_min = sigma_min, sigma_max = sigma_max_2D):

    t = torch.ones(k)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    weights = np.ones(k)/k
    xs = []
    init_x = torch.randn(k, 2) * marginal_prob_std(t)[:, None]
    xs.append(init_x)

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
                
                departure_from_uniform = np.sum(abs(k*weights-1))
                if departure_from_uniform > 0.75*k:
                    print(idx, "resampling, departure=%0.02f"%departure_from_uniform)
                
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
                                      num_steps=1000, eps=1e-3, sigma_min=sigma_min, sigma_max=sigma_max_2D, diffused_y = None):

    t = torch.ones(k)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    weights = np.ones(k)/k
    xs = []
    init_x = torch.randn(k, 2) * marginal_prob_std(t)[:, None]
    xs.append(init_x)
    if diffused_y is None:
        diffused_y = [i.repeat(k).reshape(k,1) for i in \
                      get_diffused_2D(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    else:
        diffused_y = [i.repeat(k).reshape(k,1) for i in diffused_y]
    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):
            idx = num_steps - idx - 1
            batch_time_step = torch.ones(k) * time_step
            y_obs_t = diffused_y[idx]
            x = xs[-1]
            x = torch.hstack([x,y_obs_t])[:, [0,2]]
                        
            if (idx - 1) >= 0:
                y_update_mean, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
                y_update_mean = y_update_mean[:,[1]]
                y_update_actual = diffused_y[idx-1]#.flatten()


                log_w = log_imp_weights(y_update_actual, y_update_mean, sd)

                weights *= torch.exp(log_w).cpu().detach().numpy()
                weights /= sum(weights) 
                
                #maybe keep, maybe get rid of, doesnt really matter for 2D
                departure_from_uniform = np.sum(abs(k*weights-1))
                if departure_from_uniform > 0.5*k:
                    print(idx, "resampling, departure=%0.02f"%departure_from_uniform)
                
                    resample_index = systematic(weights, k)
                    x = x[resample_index]
                    weights = np.ones_like(weights)/k
                
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
            
            batch_time_step = torch.ones(batch_size) * time_step
            g = diffusion_coeff(batch_time_step)
            batch_time_step_ = torch.reshape(batch_time_step, (x.shape[0], 1))
            x_with_t = torch.hstack([x, batch_time_step_])

            score = score_model(x_with_t)
            mean_x = x + (g**2)[:, None] * score * step_size
            sd = torch.sqrt(step_size) * g[:, None]
            
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
                           eps=1e-5, sigma_min=sigma_min, sigma_max=sigma_max_BOD, diffused_y=None):

    t = torch.ones(batch_size)
    x = torch.randn(batch_size, 7) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]

    if diffused_y is None:
        diffused_y = [i.repeat(batch_size).reshape(batch_size,5) for i in \
                      get_diffused_BOD(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    else:
        diffused_y = [i.repeat(batch_size).reshape(batch_size,5) for i in diffused_y]

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
               eps=1e-5, sigma_min=sigma_min, sigma_max=sigma_max_BOD, diffused_y=None):

    t = torch.ones(batch_size)
    init_x = torch.randn(batch_size, 7) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    if diffused_y is None:
        diffused_y = [i.repeat(batch_size).reshape(batch_size,5) for i in \
                      get_diffused_BOD(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    else:
        diffused_y = [i.repeat(batch_size).reshape(batch_size,5) for i in diffused_y]
        
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


def log_imp_weights(sample, mean, sd):
    log_w = -(1./2)*(sample-mean)**2/(sd**2)
    log_w = torch.sum(log_w, axis=[1])
    log_w -= torch.logsumexp(log_w, 0)
    return log_w

def SMCDiff_Euler_Maruyama_sampler_BOD(score_model, marginal_prob_std, diffusion_coeff, y_obs, k, num_steps=1000, 
                                       eps=1e-5, snr=signal_to_noise_ratio, sigma_min=sigma_min, sigma_max=sigma_max_BOD, diffused_y=None):

    t = torch.ones(k)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    weights = np.ones(k)/k
    xs = []
    init_x = torch.randn(k, 7) * marginal_prob_std(t)[:, None]
    xs.append(init_x)

    if diffused_y is None:
        diffused_y = [i.repeat(k).reshape(k,5) for i in get_diffused_BOD(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    else:
        diffused_y = [i.repeat(k).reshape(k,5) for i in diffused_y]
    
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


                log_w = log_imp_weights(y_update_actual, y_update_mean, sd)

                weights *= torch.exp(log_w).cpu().detach().numpy()
                weights /= sum(weights) 
                
                departure_from_uniform = np.sum(abs(k*weights-1))
                if departure_from_uniform > 0.5*k:
                    #print(idx, "resampling, departure=%0.02f"%departure_from_uniform)
                    resample_index = systematic(weights, k)
                    x = x[resample_index]
                    weights = np.ones_like(weights)/k

            mu, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
            x_t_1 = mu + sd * torch.randn_like(x) 
            xs.append(x_t_1)

    return xs[-1]

def SMCDiff_pc_sampler_BOD(score_model, marginal_prob_std, diffusion_coeff, y_obs, k, num_steps=1000, 
                           eps=1e-5, snr=signal_to_noise_ratio, sigma_min=sigma_min, sigma_max=sigma_max_BOD, diffused_y=None):

    t = torch.ones(k)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    weights = np.ones(k)/k
    xs = []
    init_x = torch.randn(k, 7) * marginal_prob_std(t)[:, None]
    xs.append(init_x)
    if diffused_y is None:
        
        diffused_y = [i.repeat(k).reshape(k,5) for i in get_diffused_BOD(y_obs, num_steps, sde_VE, sigma_min, sigma_max)]
    else:
        diffused_y = [i.repeat(k).reshape(k,5) for i in diffused_y]
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
                weights *= torch.exp(log_w).cpu().detach().numpy()
                weights /= sum(weights) # Re-normalize
                
                departure_from_uniform = np.sum(abs(k*weights-1))
                if departure_from_uniform > 0.5*k:
                    #print(idx, "resampling, departure=%0.02f"%departure_from_uniform)
                    resample_index = systematic(weights, k)
                    x = x[resample_index]
                    weights = np.ones_like(weights)/k
                
            
            mu, sd = get_next_x(x, k, score_model, diffusion_coeff, time_step, step_size)
            x_t_1 = mu + sd * torch.randn_like(x) 
            xs.append(x_t_1)

    return xs[-1]

def residual(W):

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

    su = (np.random.rand(1) + np.arange(M)) / M
    return inverse_cdf(su, W)

def multinomial(W, M):

    return inverse_cdf(uniform_spacings(M), W)

def uniform_spacings(N):

    z = np.cumsum(-np.log(np.random.rand(N + 1)))
    return z[:-1] / z[-1]

def inverse_cdf(su, W):

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




#################
#################
#################


def Euler_Maruyama_sampler_MNIST(score_model, marginal_prob_std, diffusion_coeff, 
                                 batch_size=64, num_steps=1000, device='cpu', eps=1e-3):
    
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

    return mean_x


def pc_sampler_MNIST(score_model, marginal_prob_std, diffusion_coeff, 
                     batch_size=64, num_steps=1000, snr=0.16, device='cpu', eps=1e-3):

    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)
            
    return x_mean


def get_diffused_MNIST(obs, n, sde, sigma_min=0.01, sigma_max=25):
    data = obs.clone().detach()
    t = 1e-5
    dt = 1/n
    diffused = [data.clone().detach()]
    for i in range(n):
        drift, diffusion = sde(data, t, sigma_min, sigma_max)
        data += drift * dt
        data += diffusion * torch.randn(28,28, device=device) * np.sqrt(dt)
        diffused.append(data.clone().detach())
        t += dt
        
    return torch.stack(diffused)

def insert_condition(x, y_obs):
    inserted = x.clone().detach()

    for i in inserted:
        i[0][:, :14] = y_obs

    return inserted

device='cpu'
def CDiffE_Euler_Maruyama_sampler_MNIST(score_model, marginal_prob_std, diffusion_coeff, y_obs, batch_size=16, num_steps=1000,
                                        eps=1e-3, sigma_min=sigma_min, sigma_max=sigma_max_MNIST, diffused_y=None):

    t = torch.ones(batch_size, device=device)
    x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(1)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]

    if diffused_y is None:
        diffused_y = get_diffused_MNIST(y_obs, 1000, sde_VE, sigma_min, sigma_max)
        diffused_y = [i[:,:14] for i in diffused_y]
    
    else:
        diffused_y = [i[:,:14] for i in diffused_y]

    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):
            idx = num_steps - idx - 1
            y_obs_t = diffused_y[idx]
            x = insert_condition(x, y_obs_t)
            
            batch_time_step = torch.ones(batch_size, device=device) * time_step

            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

    return mean_x

def CDiffE_pc_sampler_MNIST(score_model, marginal_prob_std, diffusion_coeff, y_obs, batch_size=16, num_steps=1000, 
                           eps=1e-3, sigma_min=sigma_min, sigma_max=sigma_max_MNIST, diffused_y=None, snr=0.16):

    t = torch.ones(batch_size, device=device)
    x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(1)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]

    if diffused_y is None:
        diffused_y = get_diffused_MNIST(y_obs, 1000, sde_VE, sigma_min, sigma_max)
        diffused_y = [i[:,:14] for i in diffused_y]
    
    else:
        diffused_y = [i[:,:14] for i in diffused_y]

    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):
            idx = num_steps - idx - 1
            y_obs_t = diffused_y[idx]
            x = insert_condition(x, y_obs_t)
            batch_time_step = torch.ones(batch_size) * time_step
            
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            x = insert_condition(x, y_obs_t)
            # Predictor step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

    return mean_x

def get_y(x, k):
    y = torch.zeros(k,1,28,14)
    for i in range(k):
        y[i][0] = x[i][0][:, :14]
        
    return y    

def log_imp_weights_MNIST(sample, mean, sd):
    log_w = -(1./2)*(sample-mean)**2/(sd**2)
    log_w = torch.sum(log_w, axis=[1,2,3])
    log_w -= torch.logsumexp(log_w, 0)
    return log_w

def SMCDiff_Euler_Maruyama_sampler_MNIST(score_model, marginal_prob_std, diffusion_coeff, y_obs, k, num_steps=1000, 
                                       eps=1e-3, sigma_min=sigma_min, sigma_max=sigma_max_MNIST, diffused_y=None):

    t = torch.ones(k, device=device)
    x = torch.randn(k, 1, 28, 28, device=device) * marginal_prob_std(1)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    weights = np.ones(k)/k
    
    if diffused_y is None:
        diffused_y = get_diffused_MNIST(y_obs, 1000, sde_VE, sigma_min, sigma_max)
        diffused_y = [i[:,:14] for i in diffused_y]
    
    else:
        diffused_y = [i[:,:14] for i in diffused_y]

    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):

            idx = num_steps - idx - 1
            y_obs_t = diffused_y[idx]
            x = insert_condition(x, y_obs_t)
            batch_time_step = torch.ones(k, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            
            if (idx - 1) >= 0:
                mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
                sd = torch.sqrt(step_size) * g[:, None, None, None]

                y_update_mean = get_y(mean_x, k)
                y_update_actual = diffused_y[idx-1]


                log_w = log_imp_weights_MNIST(y_update_actual, y_update_mean, sd)
                weights *= torch.exp(log_w).cpu().detach().numpy()
                weights /= sum(weights) 
                
                departure_from_uniform = np.sum(abs(k*weights-1))
                if (departure_from_uniform > 0.75*k) and (idx>50):
                    #print(idx, "resampling, departure=%0.02f"%departure_from_uniform)
                    resample_index = systematic(weights, k)
                    x = x[resample_index]
                    weights = np.ones_like(weights)/k
            
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
            
    return mean_x

def SMCDiff_pc_sampler_MNIST(score_model, marginal_prob_std, diffusion_coeff, y_obs, k, num_steps=1000, 
                                       eps=1e-3, sigma_min=sigma_min, sigma_max=sigma_max_MNIST, diffused_y=None, snr=0.16):

    t = torch.ones(k, device=device)
    x = torch.randn(k, 1, 28, 28, device=device) * marginal_prob_std(1)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    weights = np.ones(k)/k
    
    if diffused_y is None:
        diffused_y = get_diffused_MNIST(y_obs, 1000, sde_VE, sigma_min, sigma_max)
        diffused_y = [i[:,:14] for i in diffused_y]
    
    else:
        diffused_y = [i[:,:14] for i in diffused_y]

    with torch.no_grad():
        for idx, time_step in enumerate(notebook.tqdm(time_steps)):

            idx = num_steps - idx - 1
            y_obs_t = diffused_y[idx]
            x = insert_condition(x, y_obs_t)
            batch_time_step = torch.ones(k, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            
            # SMC step
            if (idx - 1) >= 0:
                mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
                sd = torch.sqrt(step_size) * g[:, None, None, None]

                y_update_mean = get_y(mean_x, k)
                y_update_actual = diffused_y[idx-1]


                log_w = log_imp_weights_MNIST(y_update_actual, y_update_mean, sd)
                weights *= torch.exp(log_w).cpu().detach().numpy()
                weights /= sum(weights) 
                
                departure_from_uniform = np.sum(abs(k*weights-1))
                if (departure_from_uniform > 0.75*k) and (idx>50):
                    #print(idx, "resampling, departure=%0.02f"%departure_from_uniform)
                    resample_index = systematic(weights, k)
                    x = x[resample_index]
                    weights = np.ones_like(weights)/k
            
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
            
            # Predictor step
            x = insert_condition(x, y_obs_t)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
            
    return mean_x

def CDE_pc_sampler_MNIST(score_model, marginal_prob_std, diffusion_coeff, y_obs, batch_size=16, num_steps=1000, 
                           eps=1e-3, snr=0.16):

    t = torch.ones(batch_size, device=device)
    x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(1)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    y_obs = y_obs[:, :14]
    
    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps):

            x = insert_condition(x, y_obs)
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
            
            x = insert_condition(x, y_obs)
            
            # Predictor step
            g = diffusion_coeff(batch_time_step)
            score = score_model(x, batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

    return mean_x

def CDE_Euler_Maruyama_sampler_MNIST(score_model, marginal_prob_std, diffusion_coeff, y_obs, 
                                     batch_size=16, num_steps=1000, eps=1e-3):

    t = torch.ones(batch_size, device=device)
    x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(1)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    y_obs = y_obs[:, :14]
    
    with torch.no_grad():
        for time_step in notebook.tqdm(time_steps):

            x = insert_condition(x, y_obs)

            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            score = score_model(x, batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

    return mean_x