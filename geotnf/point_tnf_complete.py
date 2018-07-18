import tensorflow as tf
import numpy as np

class PointTnf(object):
    def __init__(self):
        self.x='hello'

    def affPointTnf(self,theta,points):
        theta_mat = theta.reshape(-1,2,3)
        warped_points = tf.matmul(theta_mat[:,:,:2], points)
        warped_points += tf.tile(tf.expand_dims(theta_mat[:,:,2],dim=2)
                                 ,tf.shape(warped_points)) #translation section in Affine transformation
        return warped_points

def PointsToUnitCoords(P, im_size):
    h, w = im_size[:, 0], im_size[:, 1]
    NormAxis = lambda x, L: (x - 1 - (L - 1) / 2) * 2 / (L - 1)
    P_norm = tf.identity(P)
    # normalize Y
    P_norm[:, 0, :] = NormAxis(P[:, 0, :], tf.tile(tf.expand_dims(w, dim=1), tf.shape(P[:, 0, :])))
    # normalize X
    P_norm[:, 1, :] = NormAxis(P[:, 1, :], tf.tile(tf.expand_dims(h, dim=1), tf.shape(P[:, 1, :])))
    return P_norm

def PointsToPixelCoords(P, im_size):
    h, w = im_size[:, 0], im_size[:, 1]
    NormAxis = lambda x, L: x*(L-1)/2+1+(L-1)/2
    P_norm = tf.identity(P)
    # normalize Y
    P_norm[:, 0, :] = NormAxis(P[:, 0, :], tf.tile(tf.expand_dims(w, dim=1), tf.shape(P[:, 0, :])))
    # normalize X
    P_norm[:, 1, :] = NormAxis(P[:, 1, :], tf.tile(tf.expand_dims(h, dim=1), tf.shape(P[:, 1, :])))
    return P_norm



