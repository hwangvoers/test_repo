# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:17:00 2021

@author: hongwang
"""

import os
import numpy as np
import torch
import torchvision.utils as vutils

from CGAN_GD import generator as cganG
from CGAN_GD import Discriminator as cganD

class Model(object):
    def __init__(self, name, device, data_loader, classes, channels, img_size,
                 latent_dim):
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        if self.name == 'cgan':
            self.netG=cganG(self.classes, self.channels, self.img_size, self.latent_dim)
        self.netG.to(self.device)
        
        if self.name == 'cgan':
            self.netD = cganD(self.classes, self.channels, self.img_size, self.latent_dim)
        self.netD.to(self.device)
        self.optim_G = None
        self.optim_D = None
        
        
    def create_optim(self, lr, alpha =0.5, beta = 0.999):
        self.optim_G = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                    self.netG.parameters()),
                        lr=lr, betas = (alpha, beta))
        self.optim_D = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    self.netD.parameters()),
                        lr =lr, betas = (alpha, beta))
        

    def train (self, epochs, log_interval = 100, out_dir = '', verbose = True):
        
        self.netG.train()
        self.netD.train()
        viz_noise = torch.randn(self.data_loader.batch_size,self.latent_dim,
                                device=self.device)
        viz_label = torch.LongTensor(np.array([num for _ in range(nrows) 
                                               for num in range(8)])).to(self.device)    
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)
                real_label = torch.full((batch_size, 1), 1., device = self.device)
                fake_label = torch.full((batch_size, 1), 0., device = self.device)
                
                # Train G
                self.netG.zero_grad()
                z_noise = torch.randn(batch_size, self.latent_dim, device = self.device)
                x_fake_labels = torch.randint(0, self.classes, (batch_size,), device=self.device)
                x_fake = self.netG(z_noise, x_fake_labels)
                y_fake_g = self.netD(x_fake, x_fake_labels)
                g_loss = self.netD.loss(y_fake_g, real_label)
                g_loss.backward()
                self.optim_G.step()
                
                # Train D
                self.netD.zero_grad()
                y_real = self.netD(data, target)
                d_real_loss = self.netD.loss(y_real, real_label)
                
                y_fake_d = self.netD(x_fake.detach(), x_fake_labels)
                d_fake_loss = self.netD.loss(y_fake_d, fake_label)
                d_loss = (d_real_loss+d_fake_loss)/2
                d_loss.backward()
                self.optim_D.step()
            
            if verbose and batch_idx % log_interval == 0 and batch_idx>0:
                print('Epoch{} [{}/{}] loss_D:{:.4f} loss_G:{:.4f}'.format(
                      epoch, batch_idx, len(self.data_loader),
                              d_loss.mean().item(),
                              g_loss.mean().item()))
                vutils.save_image(data, os.path.join(out_dir,'real_samples.png'),
                                  normalize = True)
                with torch.no_grad():
                    viz_sample = self.netG(viz_noise, viz_label)
                    vutils.save_image(viz_sample, os.path.join(out_dir, 
                    'fake_samples_{}.png'.format(epoch)), nrow=8, normalize =True)
                    
                    
        self.save_to(path=out_dir, name=self.name, verbose=False)            
            
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        