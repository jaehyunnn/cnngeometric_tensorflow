import os
import sys
from skimage import io, transform
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class GeometricTnf:
    def __init__(self, geometric_model='affine', out_h=240, out_w=240):
        self.out_h = out_h
        self.out_w = out_w
        if geometric_model=='affine':
            self.gridGen = AffineGridGen(out_h, out_w)
        self.theta_identity = np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype('float32')

    def __call__(self, image_batch, theta_batch=None, padding_factor=1.0, crop_factor=1.0):
        B, H, W, C = image_batch.shape
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = np.tile(theta_batch, [B,1,1])

        sampling_grid = self.gridGen(theta_batch) * padding_factor * crop_factor

        x_s = sampling_grid[:, :, :, 0:1].squeeze()
        y_s = sampling_grid[:, :, :, 1:2].squeeze()

        # sample transformed image
        warped_image_batch = self.bilinear_sampler(image_batch, x_s, y_s)

        return warped_image_batch

    def bilinear_sampler(self, im, x_s, y_s):
        b, h, w, c = np.shape(im)

        x = ((x_s + 1.) * w) * 0.5
        y = ((y_s + 1.) * h) * 0.5

        # grab 4 nearest corner points for each (x_i, y_i)
        x0 = np.floor(x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(y).astype(np.int32)
        y1 = y0 + 1

        # make sure it's inside img range [0, H] or [0, W]
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)

        # look up pixel values at corner coords
        Ia = im[np.arange(b)[:, None, None], y0, x0]
        Ib = im[np.arange(b)[:, None, None], y1, x0]
        Ic = im[np.arange(b)[:, None, None], y0, x1]
        Id = im[np.arange(b)[:, None, None], y1, x1]

        # calculate deltas
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # add dimension for addition
        wa = np.expand_dims(wa, axis=3)
        wb = np.expand_dims(wb, axis=3)
        wc = np.expand_dims(wc, axis=3)
        wd = np.expand_dims(wd, axis=3)

        # compute output
        out = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return np.float32(out)

class SynthPairTnf:
    def __init__(self,geometric_model='affine', crop_factor=9/16, output_size=(240,240), padding_factor = 0.5):
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size
        self.rescalingTnf = GeometricTnf('affine', self.out_h, self.out_w)
        self.geometricTnf = GeometricTnf(geometric_model, self.out_h, self.out_w)

    def __call__(self, batch):
        image_batch, theta_batch = batch['image'], batch['theta']
        try:
            B, H, W, C = image_batch.shape
        except:
            image_batch = np.expand_dims(image_batch, 0)
            B, H, W, C = image_batch.shape
            theta_batch = np.expand_dims(theta_batch, 0)

        # generate symmetrically padded image for bigger sampling region (padded image)
        image_batch = self.symmetricImagePad(image_batch, self.padding_factor)

        # get cropped image (source)
        cropped_image_batch = self.rescalingTnf(image_batch, None, padding_factor=self.padding_factor, crop_factor=self.crop_factor)

        # get transformed image (target)
        warped_image_batch = self.geometricTnf(image_batch, theta_batch, padding_factor=self.padding_factor, crop_factor=self.crop_factor)

        """
        N_subplots = 3
        fig, axs = plt.subplots(1, N_subplots)
        axs[0].imshow(image_batch[0])
        axs[0].set_title('padded')
        axs[1].imshow(cropped_image_batch[0])
        axs[1].set_title('cropped')
        axs[2].imshow(warped_image_batch[0])
        axs[2].set_title('warped')
        plt.show()
        """

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch}

    def symmetricImagePad(self, image_batch, padding_factor):
        try:
            B, H, W, C = image_batch.shape
        except:
            image_batch = np.expand_dims(image_batch, 0)
            B, H, W, C = image_batch.shape

        pad_arg = ((int(H*padding_factor), int(H*padding_factor)), (int(W*padding_factor), int(W*padding_factor)))

        temp_c1 = np.expand_dims(np.expand_dims(np.pad(image_batch[0,:,:,0], pad_arg, "symmetric"),axis=0),axis=3)
        temp_c2 = np.expand_dims(np.expand_dims(np.pad(image_batch[0, :, :, 1], pad_arg, "symmetric"),axis=0),axis=3)
        temp_c3 = np.expand_dims(np.expand_dims(np.pad(image_batch[0, :, :, 2], pad_arg, "symmetric"),axis=0),axis=3)
        temp_c_concat = np.concatenate((temp_c1,temp_c2,temp_c3),3)
        temp_b = temp_c_concat

        for i in range(1,B):
            temp_c1 = np.expand_dims(np.expand_dims(np.pad(image_batch[i, :, :, 0], pad_arg, "symmetric"), axis=0), axis=3)
            temp_c2 = np.expand_dims(np.expand_dims(np.pad(image_batch[i, :, :, 1], pad_arg, "symmetric"), axis=0), axis=3)
            temp_c3 = np.expand_dims(np.expand_dims(np.pad(image_batch[i, :, :, 2], pad_arg, "symmetric"), axis=0), axis=3)

            temp_c_concat = np.concatenate((temp_c1, temp_c2, temp_c3), axis=3)
            temp_b = np.concatenate((temp_b,temp_c_concat),axis=0)
        image_batch = temp_b

        return image_batch

class AffineGridGen:
    def __init__(self, out_h=240, out_w=240, out_ch=3):
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def __call__(self, theta):
        try:
            batch_size, row_1, row_2 = theta.shape
        except:
            theta = np.expand_dims(theta, 0)
            batch_size, row_1, row_2 = theta.shape

        # create normalized 2D grid
        x = np.expand_dims(np.linspace(-1.0, 1.0, self.out_w), 1)
        y = np.linspace(-1.0, 1.0, self.out_h)
        x_t, y_t = np.meshgrid(x, y)

        # reshape to homogeneous form [x_t, y_t, 1]
        ones = np.ones(np.prod(x_t.shape))
        sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

        # repeat grid num_batch times
        sampling_grid = np.resize(sampling_grid, (batch_size, self.out_ch, self.out_h*self.out_w))

        # transform the sampling grid i.e. batch multiply
        batch_grids = np.matmul(theta, sampling_grid)
        # batch grid has shape (num_batch, 2, H*W)

        # reshape to (num_batch, height, width, 2)
        batch_grids = batch_grids.reshape(batch_size, 2, self.out_h, self.out_w)
        batch_grids = np.moveaxis(batch_grids, 1, -1)

        return batch_grids

