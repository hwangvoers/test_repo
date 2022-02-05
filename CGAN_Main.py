# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:22:07 2021

@author: hongwang
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import utils

from CGAN_Build import Model

FLAGS = None

def main ():
    device = torch.device("cuda:0" if FLAGS.cuda else "cpu")
    
    if FLAGS.train:
        print('Loading data....\n')
        dataset = dset.MNIST(root=FLAGS.data_dir, download=True,
                             transform = transforms.Compose([
                             transforms.Resize(FLAGS.img_size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,),(0.5,))
                             ]))
        assert dataset
        dataloader = torch.utils.data.Dataloader(dataset, batch_size=
                                                 FLAGS.batch_size,
                                            shuffle = True, num_workers =4,
                                            pin_memery = True)
        print('Creating model...\n')
        model = Model(FLAGS.model, device, dataloader, FLAGS.classes, FLAGS.channels,
                      FLAGS.img_size, FLAGS.latent_dim)
        model.create_optim(FLAGS.lr)
        
        #Train
        model.train(FLAGS.epochs, FLAGS.log_interval, FLAGS.out_dir, True)
        
        model.save_to('')
    else:
        model = Model(FLAGS.model, device, None, FLAGS.classes, FLAGS.channels, 
                    FLAGS.img_size, FLAGS.latent_dim)
        model.load_from(FLAGS.out_dir)
        model.eval(mode=0, batch_size = FLAGS.batch_size)
        
        

if __name__ == '__main__':
    from utils import boolean_string
    parser = argparse.ArgumentParser(description='Hands-On GANs -Chapter 5')
    parser.add_argument('--model', type=str, default='cgan', help='one of `cgan` and `infogan`.')
    parser.add_argument('--cuda', type=boolean_string, default=True, help='enable CUDA.')
    parser.add_argument('--train', type=boolean_string, default=True, help='train mode or eval mode.')
    parser.add_argument('--data_dir', type=str, default='~/Data/mnist', help='Directory for dataset.')
    parser.add_argument('--out_dir', type=str, default='output', help='Directory for output.')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='size of batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='latent space dimension')
    parser.add_argument('--classes', type=int, default=10, help='number of classes')
    parser.add_argument('--img_size', type=int, default=64, help='size of images')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--log_interval', type=int, default=100, help='interval between logging and image sampling')
    parser.add_argument('--seed', type=int, default=1, help='random seed')  
    
    FLAGS = parser.parse_args()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()