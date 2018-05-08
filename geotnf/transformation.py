import os
import sys
from skimage import io
import pandas as pd
import numpy as np
import tensorflow as tf

class GeometricTnf(object):
    def __init__(self, geometric_model='affine', out_h=240, out_w=240):
        self.out_h = out_h
        self.out_w = out_w

        if geometric_model=='affine':
            self.gridGen = AffineGridGen(out_h, out_w)

        self.theta_identity = tf.Variable(initial_value=np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype(np.float32))

    def __call__(self, image_batch, theta_batch=None, padding_factor=1.0, crop_factor=1.0):
        b, c, h, w = tf.shape(image_batch)
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = tf.tile(theta_batch, [b,2,3])
            theta_batch = tf.Variable(theta_batch, trainable=False)

        sampling_grid = self.gridGen(theta_batch)

        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data*padding_factor*crop_factor
        # sample transformed image
        warped_image_batch = #torch.nn.functional.grid_sample(image_batch, sampling_grid)

        return warped_image_batch

class SynthPairTnf(object):
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

        b, c, h, w = tf.shape(image_batch)

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
        b, c, h, w = tf.reshape(image_batch)
        pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
        idx_pad_left = tf.Variable(range(pad_w-1,-1,-1), dtype=tf.int64)
        idx_pad_right = tf.Variable(range(w-1,w-pad_w-1,-1), dtype=tf.int64)
        idx_pad_top = tf.Variable(range(pad_h-1,-1,-1), dtype=tf.int64)
        idx_pad_bottom = tf.Variable(range(h-1,h-pad_h-1,-1), dtype=tf.int64)

        image_batch = tf.concat((image_batch.index_select(3, idx_pad_left), image_batch,
                                 image_batch.index_select(3, idx_pad_right)), axis=3)
        image_batch = tf.concat((image_batch.index_select(2, idx_pad_top), image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)), axis=2)

        return image_batch

class AffineGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_ch=3):
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        batch_size = tf.shape(theta)[0]
        out_size = [batch_size, self.out_ch, self.out_h, self.out_w]
        return #torch.nn.functional.affine_grid(theta, out_size)
