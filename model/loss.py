import tensorflow as tf
import numpy as np

class TransformedGridLoss():
    def __init__(self, geometric_model='affine', grid_size=20):
        self.geometric_model = geometric_model
        axis_coords = np.linspace(-1,1,grid_size)
        self.N = grid_size*grid_size
        X, Y = np.meshgrid(axis_coords, axis_coords)
        X