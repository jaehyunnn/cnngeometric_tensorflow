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
        b, c, h, w = image_batch.size()
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = tf.tile(theta_batch, [b,2,3])

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