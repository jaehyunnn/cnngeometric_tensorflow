import tensorflow as tf
import numpy as np
from geotnf.point_tnf import PointTnf

class TransformedGridLoss():
    def __init__(self, geometric_model='affine', grid_size=20):
        self.geometric_model = geometric_model
        axis_coords = np.linspace(-1,1,grid_size)
        self.N = grid_size*grid_size
        X, Y = np.meshgrid(axis_coords, axis_coords)
        X = np.reshape(X, [1,1,self.N])
        Y = np.reshape(Y, [1,1,self.N])
        P = np.concatenate((X,Y),1)
        self.P = P
        self.pointTnf = PointTnf()

    def __call__(self, theta, theta_GT, batch_size):
        P = tf.cast(tf.tile(self.P, [batch_size,1,1]),'float32')

        if self.geometric_model == "affine":
            P_prime = self.pointTnf.affPointTnf(theta, P)
            P_prime_GT = self.pointTnf.affPointTnf(theta_GT, P)
        else:
            print("Sorry, Cannot use TPS transformation not yet")

        loss = tf.reduce_sum(tf.pow(P_prime-P_prime_GT,2),1) # Squared distance (MSE loss)
        loss = tf.reduce_mean(loss)

        return loss