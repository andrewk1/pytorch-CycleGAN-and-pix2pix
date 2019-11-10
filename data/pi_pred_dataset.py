import os.path
import numpy as np
from PIL import Image, ImageMath
import torchvision.transforms as transforms
import torch
import random

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class PiPredDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_canonical = os.path.join(opt.dataroot, opt.phase + 'canonical') 

        self.canonical_pi = np.load(os.path.join(opt.dataroot, opt.phase + '_canonical_pi.npy'))
        self.canonical_paths = make_dataset(self.dir_canonical, opt.max_dataset_size)

        self.canonical_size = len(self.canonical_paths)

        self.transform_rgb = get_transform(self.opt, grayscale=False)
        self.transform_grayscale = get_transform(self.opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, canonical_paths and random_paths 
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            canonical_paths (str)    -- image paths
            random_paths (str)    -- image paths
        """
        canonical_path = self.canonical_paths[index % self.canonical_size]  # make sure index is within then range
        seed = int(canonical_path[:canonical_paths.find('img') - 1])
        index = int(canonical_path[canonical_paths.find('img') + 3:])
        print(seed)
        print(index)

        canonical_pi = self.canonical_pi[index]
        canonical_img = Image.open(canonical_path).convert('RGB')
    
        # apply image transformation
        canonical = self.transform_rgb(canonical_img)

        return {'canonical': canonical, 'canonical_pi': canonical_pi}

    def __len__(self):
        """
        Return the total number of images in the dataset (real + randomly generated)
        """
        return self.canonical_size
