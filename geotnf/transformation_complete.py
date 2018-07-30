import os
import sys
from skimage import io
import pandas as pd
import numpy as np
import tensorflow as tf

class GeometricTnf:
    def __init__(self, geometric_model='affine', out_h=240, out_w=240):
        self.out_h = out_h
        self.out_w = out_w

        if geometric_model=='affine':
            self.gridGen = AffineGridGen(out_h, out_w)

        self.theta_identity = tf.Variable(initial_value=np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype(np.float32))

    def __call__(self, image_batch, theta_batch=None, padding_factor=1.0, crop_factor=1.0):
        B, H, W, C = image_batch.get_shape().as_list()
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = tf.tile(theta_batch, [B,1,1])
            theta_batch = tf.Variable(theta_batch, trainable=False)

        sampling_grid = self.gridGen(theta_batch)

        x_s = sampling_grid[:, 0, :, :]*padding_factor*crop_factor  # transform 된 x좌표
        #x_s = tf.tile(tf.expand_dims(x_s,3), [1, 1, 1, C])
        y_s = sampling_grid[:, 1, :, :]*padding_factor*crop_factor  # transform 된 y좌표
        #y_s = tf.tile(tf.expand_dims(y_s,3), [1, 1, 1, C])
        """
        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data*padding_factor*crop_factor
        """
        # sample transformed image
        warped_image_batch = self.bilinear_sampler(image_batch, x_s, y_s)

        return warped_image_batch

    def get_pixel_value(self, img, x, y):
        B, H, W, C = img.get_shape().as_list()
        batch_idx = tf.range(0, B)
        batch_idx = tf.reshape(batch_idx, (B, 1, 1))  # axis 1,2에 expand_dims 한것과 같음
        b = tf.tile(batch_idx, (1, x.shape[1], x.shape[2]))

        indices = tf.stack([b, y, x], 3)
        pixel_value = tf.gather_nd(img, indices)

        return pixel_value

    def bilinear_sampler(self, img, x, y):
        B, H, W, C = img.get_shape().as_list()

        max_y = tf.cast(H - 1, 'int32')
        max_x = tf.cast(W - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # rescale x and y to [0, W-1 or H-1]
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        x = 0.5 * ((x + 1) * tf.cast(max_x - 1, 'float32'))
        y = 0.5 * ((y + 1) * tf.cast(max_y - 1, 'float32'))

        # grab 4 nearest corner points for each (x_i, y_i)
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        # clip to range [0, H-1 or W-1] to not violate img boundaries
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        """ min보다 작으면 min으로 max보다 크면 max로 잘라버림
        모서리 부분은 nearest corner가 범위를 벗어 날수 있으므로..."""

        # get pixel value at corner coords
        Ia = self.get_pixel_value(img, x0, y0)
        Ib = self.get_pixel_value(img, x0, y1)
        Ic = self.get_pixel_value(img, x1, y0)
        Id = self.get_pixel_value(img, x1, y1)
        # recast as float for delta calculation
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        # calculate deltas
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        wa = tf.tile(tf.expand_dims(wa, 3), [1, 1, 1, C])
        wb = tf.tile(tf.expand_dims(wb, 3), [1, 1, 1, C])
        wc = tf.tile(tf.expand_dims(wc, 3), [1, 1, 1, C])
        wd = tf.tile(tf.expand_dims(wd, 3), [1, 1, 1, C])

        # compute output
        out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

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
            B, H, W, C = image_batch.get_shape().as_list()
        except:
            B, H, W, C = tf.expand_dims(image_batch, 0).get_shape().as_list()

        # generate symmetrically padded image for bigger sampling region
        image_batch = self.symmetricImagePad(image_batch, self.padding_factor)

        # convert to variables
        image_batch = tf.Variable(image_batch, trainable=False)
        theta_batch = tf.Variable(theta_batch, trainable=False)

        # get cropped image
        cropped_image_batch = self.rescalingTnf(image_batch, None, self.padding_factor, self.crop_factor)

        # get transformed image
        warped_image_batch = self.geometricTnf(image_batch, theta_batch,
                                               self.padding_factor, self.crop_factor)

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch}

    def symmetricImagePad(self, image_batch, padding_factor):
        try:
            B, H, W, C = image_batch.get_shape().as_list()
        except:
            B, H, W, C = tf.expand_dims(image_batch,0).get_shape().as_list()
        pad_h, pad_w = int(H * padding_factor), int(W * padding_factor)
        idx_pad_left = tf.range(pad_w-1, -1, -1, dtype=tf.int64, name='idx_pad_left')
        idx_pad_right = tf.range(W-1, W-pad_w-1, -1, dtype=tf.int64, name='idx_pad_right')
        idx_pad_top = tf.range(pad_h-1, -1, -1, dtype=tf.int64, name='idx_pad_top')
        idx_pad_bottom = tf.range(H-1, H-pad_h-1, -1, dtype=tf.int64, name='idx_pad_bottom')

        image_batch = tf.concat((image_batch.index_select(3, idx_pad_left), image_batch,
                                 image_batch.index_select(3, idx_pad_right)), axis=3)
        image_batch = tf.concat((image_batch.index_select(2, idx_pad_top), image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)), axis=2)

        return image_batch

class AffineGridGen:
    def __init__(self, out_h=240, out_w=240, out_ch=3):
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def __call__(self, theta):
        batch_size = theta.get_shape().as_list()[0]
        out_size = [batch_size, self.out_h, self.out_w, self.out_ch]

        # create normalized 2D grid
        x = tf.linspace(-1.0, 1.0, self.out_w)
        y = tf.linspace(-1.0, 1.0, self.out_h)
        x_t, y_t = tf.meshgrid(x, y)

        # flatten
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])

        # reshape to homogeneous form [x_t, y_t, 1]
        ones = tf.ones_like(x_t_flat)
        sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

        # repeat grid batch_size times
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
        sampling_grid = tf.tile(sampling_grid, [batch_size, 1, 1])

        # cast to float32 (required for matmul)
        theta = tf.cast(theta, 'float32')
        sampling_grid = tf.cast(sampling_grid, 'float32')

        # transform the sampling grid - batch multiply
        batch_grids = tf.matmul(theta, sampling_grid)  # batch_grids 는 transform 된 좌표값
        # batch grid has shape (batch_size, 2, H*W)

        # reshape to (batch_size, H, W, 2)
        batch_grids = tf.reshape(batch_grids, [batch_size, 2, self.out_h, self.out_w])
        return batch_grids


