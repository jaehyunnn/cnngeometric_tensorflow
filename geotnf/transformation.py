import os
import sys
from skimage import io, transform
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class GeometricTnf:
    def __init__(self, geometric_model='affine', out_h=240, out_w=240, resize=False):
        self.out_h = out_h
        self.out_w = out_w
        self.resize = resize
        if geometric_model=='affine':
            self.gridGen = AffineGridGen(out_h, out_w)

        self.theta_identity = np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype('float32')

    def __call__(self, image_batch, theta_batch=None, padding_factor=1.0, crop_factor=1.0):
        B, H, W, C = image_batch.shape
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = np.tile(theta_batch, [B,1,1])

        if self.resize:
            # Resize
            if image_batch.dtype == 'uint8':
                image_batch = np.float32(image_batch)/255.
                image_batch = transform.resize(image_batch, [B, self.out_h, self.out_w, C])
            elif image_batch.dtype == 'float32':
                try:
                    image_batch = transform.resize(image_batch, [B, self.out_h, self.out_w, C])
                except:
                    image_batch = transform.resize(np.uint8(image_batch), [B, self.out_h, self.out_w, C])
                    image_batch = np.float32(image_batch)/255.

        sampling_grid = self.gridGen(theta_batch)

        x_s = sampling_grid[:, :, :, 0:1].squeeze()
        y_s = sampling_grid[:, :, :, 1:2].squeeze()

        x = ((x_s + 1.) * self.out_w) * 0.5 * padding_factor * crop_factor
        y = ((y_s + 1.) * self.out_h) * 0.5 * padding_factor * crop_factor

        # sample transformed image
        warped_image_batch = self.bilinear_sampler(image_batch, x, y)
        return warped_image_batch


    def bilinear_sampler(self, img, x, y):
        B, H, W, C = img.shape
        x0 = np.floor(x).astype(np.int64)
        x1 = x0 + 1
        y0 = np.floor(y).astype(np.int64)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, self.out_w - 1)
        x1 = np.clip(x1, 0, self.out_w - 1)
        y0 = np.clip(y0, 0, self.out_h - 1)
        y1 = np.clip(y1, 0, self.out_h - 1)

        Ia = img[np.arange(B)[:, None, None], y0, x0]
        Ib = img[np.arange(B)[:, None, None], y1, x0]
        Ic = img[np.arange(B)[:, None, None], y0, x1]
        Id = img[np.arange(B)[:, None, None], y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        wa = np.expand_dims(wa, axis=3)
        wb = np.expand_dims(wb, axis=3)
        wc = np.expand_dims(wc, axis=3)
        wd = np.expand_dims(wd, axis=3)

        out = wa * Ia + wb * Ib + wc * Ic + wd * Id

        if out.dtype == 'uint8':
            out = np.float32(out)/255.

        return out

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

        # generate symmetrically padded image for bigger sampling region
        image_batch = self.symmetricImagePad(image_batch, self.padding_factor)

        # get cropped image
        cropped_image_batch = self.rescalingTnf(image_batch, None, self.padding_factor,self.crop_factor)

        # get transformed image
        warped_image_batch = self.geometricTnf(image_batch, theta_batch, self.padding_factor, self.crop_factor)

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch}

    def symmetricImagePad(self, image_batch, padding_factor):
        try:
            B, H, W, C = image_batch.shape
        except:
            image_batch = np.expand_dims(image_batch, 0)
            B, H, W, C = image_batch.shape

        pad_arg = ((240, 240), (320, 320))

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
        x = np.linspace(-1.0, 1.0, self.out_w)
        y = np.linspace(-1.0, 1.0, self.out_h)
        x_t, y_t = np.meshgrid(x, y)

        # reshape to homogeneous form [x_t, y_t, 1]
        ones = np.ones(np.prod(x_t.shape))
        sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

        # repeat grid num_batch times
        sampling_grid = np.resize(sampling_grid, (batch_size, 3, self.out_h*self.out_w))

        # transform the sampling grid i.e. batch multiply
        batch_grids = np.matmul(theta, sampling_grid)
        # batch grid has shape (num_batch, 2, H*W)

        # reshape to (num_batch, height, width, 2)
        batch_grids = batch_grids.reshape(batch_size, 2, self.out_h, self.out_w)
        batch_grids = np.moveaxis(batch_grids, 1, -1)

        return batch_grids
