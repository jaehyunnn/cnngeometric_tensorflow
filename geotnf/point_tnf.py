import tensorflow as tf
import numpy as np

class PointTnf:
    def __init__(self):
        pass

    def affPointTnf(self,theta,points):
        theta_mat = tf.reshape(theta, [-1,2,3])
        batch_size = theta_mat.get_shape().as_list()[0]

        warped_points = tf.matmul(theta_mat[:,:,:2], points)
        warped_points = tf.matmul(theta_mat[:, :, :2], points)
        tile_arg_1 = np.divide(np.array(warped_points.get_shape().as_list())[1],
                               np.array(tf.expand_dims(theta_mat[:, :, 2], axis=2).get_shape().as_list())[1]).astype('int32')
        tile_arg_2 = np.divide(np.array(warped_points.get_shape().as_list())[2],
                               np.array(tf.expand_dims(theta_mat[:, :, 2], axis=2).get_shape().as_list())[2]).astype('int32')

        warped_points += tf.tile(tf.expand_dims(theta_mat[:, :, 2], axis=2), [1, tile_arg_1, tile_arg_2])
        return warped_points

def PointsToUnitCoords(P, im_size):
    h, w = im_size[:, 0], im_size[:, 1]
    NormAxis = lambda x, L: (x - 1 - (L - 1) / 2) * 2 / (L - 1)
    P_norm = tf.identity(P)
    # normalize Y
    P_norm[:, 0, :] = NormAxis(P[:, 0, :], tf.tile(tf.expand_dims(w, axis=1), tf.shape(P[:, 0, :])))
    # normalize X
    P_norm[:, 1, :] = NormAxis(P[:, 1, :], tf.tile(tf.expand_dims(h, axis=1), tf.shape(P[:, 1, :])))
    return P_norm

def PointsToPixelCoords(P, im_size):
    h, w = im_size[:, 0], im_size[:, 1]
    NormAxis = lambda x, L: x*(L-1)/2+1+(L-1)/2
    P_norm = tf.identity(P)
    # normalize Y
    P_norm[:, 0, :] = NormAxis(P[:, 0, :], tf.tile(tf.expand_dims(w, axis=1), tf.shape(P[:, 0, :])))
    # normalize X
    P_norm[:, 1, :] = NormAxis(P[:, 1, :], tf.tile(tf.expand_dims(h, axis=1), tf.shape(P[:, 1, :])))
    return P_norm
