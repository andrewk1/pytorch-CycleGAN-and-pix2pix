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
        self.dir_canonical = os.path.join(opt.dataroot, opt.phase + 'canonical')  # create a path '/path/to/data/traincanonical'
        self.dir_random = os.path.join(opt.dataroot, opt.phase + 'random')  # create a path '/path/to/data/trainrandom'
        self.dir_seg = os.path.join(opt.dataroot, opt.phase + 'segmentation')  # create a path '/path/to/data/trainseg'
        self.dir_depth = os.path.join(opt.dataroot, opt.phase + 'depth')  # create a path '/path/to/data/traindepth'

        self.canonical_paths = sorted(make_dataset(self.dir_canonical, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.random_paths = sorted(make_dataset(self.dir_random, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.seg_paths = sorted(make_dataset(self.dir_seg, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.depth_paths = sorted(make_dataset(self.dir_depth, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.canonical_size = len(self.canonical_paths)  # get the size of dataset A
        self.random_size = len(self.random_paths)  # get the size of dataset B
        self.seg_size = len(self.seg_paths)  # get the size of dataset B
        self.depth_size = len(self.depth_paths)  # get the size of dataset B

        assert self.canonical_size == self.random_size == self.seg_size == self.depth_size, 'Dataset sizes are not the same'

        input_nc = self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.output_nc      # get the number of channels of output image
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
        canonical_path = self.canonical_paths[index % self.canonical_size]  # make sure index is within then range
        random_path = self.random_paths[index % self.canonical_size]
        seg_path = self.seg_paths[index % self.canonical_size]
        depth_path = self.depth_paths[index % self.canonical_size]
        #idx = seg_path.find('img')
        #idx2 = seg_path.find('segmentation/') + len('segmentation/')
        #rollout_num = int(seg_path[idx2:idx])
        #depth_path = self.depth_paths[rollout_num]
        #print('---')
        #print(canonical_path)
        #print(random_path)
        #print(seg_path)
        #print(depth_path)
        #print(int(seg_path[idx+4:-4]))
        #print('---')

        canonical_img = Image.open(canonical_path).convert('RGB')
        random_img = Image.open(random_path).convert('RGB')
        seg_img = Image.open(seg_path)
        depth_img = Image.open(depth_path)
        #depth = np.load(depth_path)[int(seg_path[idx+4:-4]) - 1]
        
        # apply image transformation
        canonical = self.transform_rgb(canonical_img)
        random = self.transform_rgb(random_img)
        seg = self.transform_grayscale(seg_img)
        depth = self.transform_grayscale(depth_img)

        return {'canonical': canonical, 'random': random, 'seg': seg, 'depth': depth,
                'canonical_path': canonical_path, 'random_path': random_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.canonical_size
