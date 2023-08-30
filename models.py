import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F



#Implimentation from shi et al - best model so far :)
class ScoreNet_2D(nn.Module):
    
    def __init__(self, marginal_prob_std):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        
        self.net = MLP(3 * 32,
                       layer_widths=[128,128] + [2],
                       activate_final = False,
                       activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(16,
                             layer_widths=[16] + [32],
                             activate_final = True,
                             activation_fn=torch.nn.LeakyReLU())

        self.xy_encoder = MLP(2,
                              layer_widths=[32] + [64],
                              activate_final = True,
                              activation_fn=torch.nn.LeakyReLU())
        
    def forward(self, x):
        t = x[:, -1]
        norm = self.marginal_prob_std(t)[:, None]
        t = t.reshape(-1,1)
        xy = x[:,[0,1]]
        
        t_emb = get_timestep_embedding(t, 16, 10000)
        t_emb = self.t_encoder(t_emb)
        xy_emb = self.xy_encoder(xy)

        
        h = torch.cat([xy_emb, t_emb], -1)

        out = self.net(h) 

        out = out / norm
        return out
    

class ScoreNet_BOD(nn.Module):
    
    def __init__(self, marginal_prob_std):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        
        self.net = MLP(3 * 32,
                       layer_widths=[128,128] + [7],
                       activate_final = False,
                       activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(16,
                             layer_widths=[16] + [32],
                             activate_final = True,
                             activation_fn=torch.nn.LeakyReLU())

        self.xy_encoder = MLP(2 + 5,
                              layer_widths=[32] + [64],
                              activate_final = True,
                              activation_fn=torch.nn.LeakyReLU())
        
    def forward(self, x):
        t = x[:, -1]
        norm = self.marginal_prob_std(t)[:, None]
        t = t.reshape(-1,1)
        xy = x[:,[0,1,2,3,4,5,6]]
        
        t_emb = get_timestep_embedding(t, 16, 10000)
        t_emb = self.t_encoder(t_emb)
        xy_emb = self.xy_encoder(xy)

        
        h = torch.cat([xy_emb, t_emb], -1)

        out = self.net(h) 

        out = out / norm
        return out
    
class cde_ScoreNet_BOD(nn.Module):
    
    def __init__(self, marginal_prob_std):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        
        self.net = MLP(3 * 32,
                       layer_widths=[128,128] + [2],
                       activate_final = False,
                       activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(16,
                             layer_widths=[16] + [32],
                             activate_final = True,
                             activation_fn=torch.nn.LeakyReLU())

        self.xy_encoder = MLP(2 + 5,
                              layer_widths=[32] + [64],
                              activate_final = True,
                              activation_fn=torch.nn.LeakyReLU())
        
    def forward(self, x):
        t = x[:, -1]
        norm = self.marginal_prob_std(t)[:, None]
        t = t.reshape(-1,1)
        xy = x[:,[0,1,2,3,4,5,6]]
        
        t_emb = get_timestep_embedding(t, 16, 10000)
        t_emb = self.t_encoder(t_emb)
        xy_emb = self.xy_encoder(xy)

        
        h = torch.cat([xy_emb, t_emb], -1)

        out = self.net(h) 

        out = out / norm
        return out
    
class cde_ScoreNet_2D(nn.Module):
    
    def __init__(self, marginal_prob_std):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        
        self.net = MLP(3 * 32,
                       layer_widths=[128,128] + [1],
                       activate_final = False,
                       activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(16,
                             layer_widths=[16] + [32],
                             activate_final = True,
                             activation_fn=torch.nn.LeakyReLU())

        self.xy_encoder = MLP(2,
                              layer_widths=[32] + [64],
                              activate_final = True,
                              activation_fn=torch.nn.LeakyReLU())
        
    def forward(self, x, y, t):

        xy = torch.hstack([x, y])
        
        t_emb = get_timestep_embedding(t, 16, 10000)
        t_emb = self.t_encoder(t_emb)
        xy_emb = self.xy_encoder(xy)

        
        h = torch.cat([xy_emb, t_emb], -1)

        out = self.net(h) 

        out = out / self.marginal_prob_std(t)

        return out
   
    
def get_timestep_embedding(timesteps, embedding_dim=128, max_period=10000):
    """
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    half_dim = embedding_dim // 2
    emb = math.log(max_period) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)

    emb = timesteps * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0,1])

    return emb

class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn=F.relu):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x
    


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class MNIST_ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))
  
    # Encoding path
    h1 = self.conv1(x)    
    ## Incorporate information from t
    h1 += self.dense1(embed)
    
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)
    

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
   
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
   
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    
    return h
  