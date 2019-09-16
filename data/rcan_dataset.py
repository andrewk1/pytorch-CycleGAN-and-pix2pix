import os.path
import numpy as np
from PIL import Image, ImageMath
import torchvision.transforms as transforms
import torch
import random

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class RCANDataset(BaseDataset):
    """
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_canonical = os.path.join(opt.dataroot, opt.phase + 'canonical') 
        self.dir_random = os.path.join(opt.dataroot, opt.phase + 'random') 
        self.dir_real = os.path.join(opt.dataroot, opt.phase + 'real') 
        self.dir_seg = os.path.join(opt.dataroot, opt.phase + 'segmentation') 
        self.dir_depth = os.path.join(opt.dataroot, opt.phase + 'depth') 

        self.canonical_paths = sorted(make_dataset(self.dir_canonical, opt.max_dataset_size))
        self.random_paths = sorted(make_dataset(self.dir_random, opt.max_dataset_size)) 
        self.real_paths = sorted(make_dataset(self.dir_random, opt.max_dataset_size)) 
        self.seg_paths = sorted(make_dataset(self.dir_seg, opt.max_dataset_size)) 
        self.depth_paths = sorted(make_dataset(self.dir_depth, opt.max_dataset_size)) 

        self.canonical_size = len(self.canonical_paths)
        self.random_size = len(self.random_paths)
        self.real_size = len(self.real_paths)
        self.seg_size = len(self.seg_paths)
        self.depth_size = len(self.depth_paths)

        # TODO: THis is commented for 256 set
        # assert self.canonical_size == self.random_size == self.seg_size == self.depth_size, 'Dataset sizes are not the same'

        self.transform_rgb = get_transform(self.opt, grayscale=False)
        self.transform_grayscale = get_transform(self.opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, canonical_paths andrandom_paths 
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            canonical_paths (str)    -- image paths
            random_paths (str)    -- image paths
        """
        if index > self.canonical_size:
            real_path = self.real_paths[index % self.canonical_size]
            real_img = Image.open(real_path).convert('RGB')
            real = self.transform_rgb(real_img)
            return {'real': real, 'real_path': real_path}
        else:
            real_path = None
            canonical_path = self.canonical_paths[index % self.canonical_size]  # make sure index is within then range
            random_path = self.random_paths[index % self.canonical_size]
            seg_path = self.seg_paths[index % self.canonical_size]
            depth_path = self.depth_paths[index % self.canonical_size]

            canonical_img = Image.open(canonical_path).convert('RGB')
            random_img = Image.open(random_path).convert('RGB')
            seg_img = Image.open(seg_path)
            depth_img = Image.open(depth_path)
        
            # apply image transformation
            canonical = self.transform_rgb(canonical_img)
            random = self.transform_rgb(random_img)
            seg = self.transform_grayscale(seg_img)
            depth = self.transform_grayscale(depth_img)

            return {'canonical': canonical, 'random': random, 'seg': seg, 'depth': depth,
                    'canonical_path': canonical_path, 'random_path': random_path, 'real': None}

    def __len__(self):
        """
        Return the total number of images in the dataset (real + randomly generated)
        """
        return self.canonical_size + self.real_size
