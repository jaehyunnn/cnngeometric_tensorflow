import tensorflow as tf
import numpy as np

class NormalizeImageDict(object):
    def __init__(self, image_keys, normalizeRange=True):
        self.image_keys = image_keys
        self.normalizeRange = normalizeRange

    def __call__(self, sample):
        for key in self.image_keys:
            if self.normalizeRange:
                sample[key] /= 255.0
            sample[key] = tf.image.per_image_standardization(sample[key])

        return sample

def normalize_image(image, forward=True, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    im_size = image.get_shape().as_list()
    mean = tf.Variable(initial_value=mean, dtype=tf.float32)
    mean = tf.expand_dims(tf.expand_dims(mean, dim=1), dim=2)
    std = tf.Variable(initial_value=std, dtype=tf.float32)
    std = tf.expand_dims(tf.expand_dims(std, dim=1), dim=2)

    if isinstance(image, tf.Variable):
        mean = tf.Variable(mean, trainable=False)
        std = tf.Variable(std, trainable=False)
    if forward:
        if len(im_size) == 3:
            tile_arg = np.divide(np.array(im_size.as_list()), np.array(mean.get_shape().as_list())).astype('int32')
            result = tf.divide(tf.subtract(image, tf.tile(mean, tile_arg)),
                               tf.tile(std, tile_arg))
        elif len(im_size) == 4:
            mean = tf.expand_dims(mean,0)
            std = tf.expand_dims(std,0)
            tile_arg = np.divide(np.array(im_size.as_list()),np.array(mean.get_shape().as_list())).astype('int32')
            result = tf.divide(tf.subtract(image,tf.tile(mean, tile_arg)),
                               tf.tile(std, tile_arg))
    else:
        if len(im_size) == 3:
            tile_arg = np.divide(np.array(im_size.as_list()), np.array(mean.get_shape().as_list())).astype('int32')
            result = tf.add(tf.multiply(image,tf.tile(mean, tile_arg)),
                            tf.tile(std, tile_arg))
        elif len(im_size) == 4:
            mean = tf.expand_dims(mean, 0)
            std = tf.expand_dims(std, 0)
            tile_arg = np.divide(np.array(im_size.as_list()), np.array(mean.get_shape().as_list())).astype('int32')
            result = tf.add(tf.multiply(image,tf.tile(mean, tile_arg)),
                            tf.tile(std, tile_arg))

    return result