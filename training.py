import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import notebook


def loss_fn(model, x, marginal_prob_std, eps=1e-5):

    random_t = torch.rand(x.shape[0]) * (1. - eps) + eps  
    std = marginal_prob_std(random_t)
    random_t = torch.reshape(random_t, (x.shape[0], 1))
    z = torch.randn_like(x)
    perturbed_x = x + z * std[:, None]
    x_with_t = torch.hstack([perturbed_x,random_t])
    x_with_t = x_with_t.to(torch.float32)
    score = model(x_with_t)
    loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=0))
    
    #losses = torch.square(score * std[:, None] + z)
    #losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    #loss = torch.mean(losses)
    return loss

def CDE_loss_fn_BOD(model, x, marginal_prob_std, eps=1e-5):

    y = x[:,[2,3,4,5,6]]
    x = x[:,[0,1]]
    random_t = torch.rand(x.shape[0]) * (1. - eps) + eps  
    std = marginal_prob_std(random_t)
    random_t = torch.reshape(random_t, (x.shape[0], 1))
    z = torch.randn_like(x)
    perturbed_x = x + z * std[:, None]
    perturbed_x = torch.hstack([perturbed_x,y])
    
    x_with_t = torch.hstack([perturbed_x,random_t])
    x_with_t = x_with_t.to(torch.float32)
    score = model(x_with_t)
    loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=0))
    
    #losses = torch.square(score * std[:, None] + z)
    #losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    #loss = torch.mean(losses)
    return loss

def CDE_loss_fn_2D(model, x, marginal_prob_std, eps=1e-5):

    y = torch.reshape(x[:,1], (x.shape[0], 1))
    random_t = torch.rand(x.shape[0], device='cpu') * (1. - eps) + eps  
    std = marginal_prob_std(random_t)
    random_t = torch.reshape(random_t, (x.shape[0], 1))
    z = torch.randn_like(x)
    perturbed_x = x + z * std[:, None]
    perturbed_x = torch.hstack([perturbed_x,y])
    perturbed_x = perturbed_x[:, [0,2]]
    
    x_with_t = torch.hstack([perturbed_x,random_t])
    x_with_t = x_with_t.to(torch.float32)
    score = model(x_with_t)
    loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=0))
    return loss


def train_model(score_model, data, loss_fn, marginal_prob_std_fn, file, epochs = 100, batch_size = 32, lr = 1e-4):
    
    optimizer = Adam(score_model.parameters(), lr=lr)
    dataset = data
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    tqdm_epoch = notebook.trange(epochs)
    losses = []
    
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x in data_loader:
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        torch.save(score_model.state_dict(), file)
        losses.append(avg_loss / num_items)
        
    return losses     
        
        
