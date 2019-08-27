import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


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
        self.dir_canonical = os.path.join(opt.dataroot, opt.phase + 'canonical')  # create a path '/path/to/data/trainA'
        self.dir_random = os.path.join(opt.dataroot, opt.phase + 'random')  # create a path '/path/to/data/trainB'
        self.dir_seg = os.path.join(opt.dataroot, opt.phase + 'seg')  # create a path '/path/to/data/trainB'

        self.canonical_paths = sorted(make_dataset(self.dir_canonical, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.random_paths = sorted(make_dataset(self.dir_random, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.seg_paths = sorted(make_dataset(self.dir_seg, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.canonical_size = len(self.canonical_paths)  # get the size of dataset A
        self.random_size = len(self.random_paths)  # get the size of dataset B
        self.seg_size = len(self.seg_paths)  # get the size of dataset B

        assert self.canonical_size == self.random_size == self.seg_size, 'Dataset sizes are not the same'

        input_nc = self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=False)
        self.transform_B = get_transform(self.opt, grayscale=False)

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
        
        canonical_img = Image.open(canonical_path).convert('RGB')
        random_img = Image.open(random_path).convert('RGB')
        seg_img = Image.open(seg_path).convert('RGB')

        # apply image transformation
        canonical = self.transform_A(canonical_img)
        random = self.transform_B(random_img)
        seg = self.transform_B(seg_img)

        return {'canonical': canonical, 'random': random, 'seg': seg}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.canonical_size
