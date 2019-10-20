"""Given two directories of images (same size), compute pairwise MSE between the two sets"""
import torch
from PIL import Image
import os
import numpy as np

def main():
   dir1 = '/home/robot/andrewk/pytorch-CycleGAN-and-pix2pix/results/wood_cycleGAN/test_latest/images/fakeA/'
   dir2 = '/home/robot/andrewk/pytorch-CycleGAN-and-pix2pix/results/wood_cycleGAN/test_latest/images/realA/'
   dir4 = '/home/robot/andrewk/pytorch-RCAN/results/rcan_extreme/test_latest/images/canonical_pred/'

   mse = 0
   mse2 = 0
   mse3 = 0 
   for im1, im2, im3, im4 in zip(os.listdir(dir1), os.listdir(dir2), os.listdir(dir3), os.listdir(dir4)):
       t1, t2, t3, t4 = torch.Tensor(np.array(Image.open(dir1 + im1))), torch.Tensor(np.array(Image.open(dir2 + im2))), torch.Tensor(np.array(Image.open(dir3 + im3))), torch.Tensor(np.array(Image.open(dir4 + im4)))
       mse += torch.nn.functional.mse_loss(t1, t2)
       mse2 += torch.nn.functional.mse_loss(t3, t2)
       mse3 += torch.nn.functional.mse_loss(t4, t2)

   print(mse)
   print(mse2)
   print(mse3)

if __name__ == '__main__':
    main()
