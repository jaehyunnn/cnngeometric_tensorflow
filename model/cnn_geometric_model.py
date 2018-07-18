import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

vgg = slim.nets.vgg
resnet_v2 = slim.nets.resnet_v2

class FeatureExtraction():
    def __init__(self, trainable=False, feature_extraction_cnn='vgg', normalization=True):
        self.normalization = normalization

        if feature_extraction_cnn == 'vgg':
            self.model = vgg

        if feature_extraction_cnn == 'resnet_v2':
            self.model = resnet_v2

    def forward(self, batch_size):
        features = self.model.ru
        if self.normalization:
            features = tf.nn.l2_normalize(features, epsilon=(1e-6))
        return features

class FeatureCorrelation():
    def __init__(self, dim='3D', normalization=True):
        self.normalization = normalization
        self.dim = dim

    def forward(self, feature_A, feature_B):
        """
        :param b: batch_size
        :param c: channel(amount of filters)
        :param h: height
        :param w: width
        :return:
        """
        b,c,h,w = feature_A.shape
        if self.dim == '3D':
            feature_A = tf.transpose(feature_A, [0,1,3,2])
            feature_A = tf.reshape(feature_A, [b,c,h*w])
            feature_B = tf.reshape(feature_B, [b,c,h*w])
            feature_B = tf.transpose(feature_B, [0,2,1])

            feature_mul = tf.matmul(feature_B,feature_A)
            correlation_tensor = tf.reshape(feature_mul, [b,h,w,h*w])
            correlation_tensor = tf.transpose(correlation_tensor, [0,2,3,1])
        elif self.dim =='4D':
            feature_A = tf.reshape(feature_A, [b,c,h*w])
            feature_A = tf.transpose(feature_A, [0,2,1])
            feature_B = tf.reshape(feature_B, [b,c,h*w])

            feature_mul = tf.matmul(feature_A, feature_B)
            correlation_tensor = tf.reshape(b,h,w,h*w)
            correlation_tensor = tf.expand_dims(correlation_tensor, dim=1)

        if self.normalization:
            correlation_tensor = tf.nn.l2_normalize(correlation_tensor, epsilon=(1e-6))

        return correlation_tensor

class FeatureRegression():
    def __init__(self, output_dim=6, batch_normalization=True, kernel_sizes=[7,5], channels=[128,64], feature_size=15):
        self.output_dim = output_dim
        self.batch_normalization = batch_normalization
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.feature_size = feature_size

    def forward(self, x):
        conv1 = tf.layers.conv2d(inputs=x, kernel_size=[self.kernel_sizes[0],self.kernel_sizes[0]], filters=self.channels[0], padding='SAME', activation=None)
        if self.batch_normalization:
            conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(inputs=conv1, kernel_size=[self.kernel_sizes[1],self.kernel_sizes[1]], filters=self.channels[1], padding='SAME', activation=None)
        if self.batch_normalization:
            conv2 = tf.layers.batch_normalization(conv2)
        conv2 = tf.nn.relu(conv2)

        flat = tf.layers.flatten(conv2)
        out = tf.layers.dense(inputs=flat, units=self.output_dim)

        return out

class CNNGeometric():
    def __init__(self, output_dim=6,
                 feature_extraction_cnn='vgg',
                 return_correlation=False,
                 fr_feature_size=15,
                 fr_kernel_sizes=[7,5],
                 fr_channels=[128,64],
                 feature_self_matching=False,
                 normalize_features=True, normalize_matches=True,
                 batch_normalization=True,
                 trainable=False):
        self.feature_self_matching = feature_self_matching
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        self.FeatureExtraction = FeatureExtraction(trainable=trainable,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   normalization=normalize_features)
        self.FeatureCorrelation = FeatureCorrelation(dim='3D', normalization=normalize_matches)
        self.FeatureRegression = FeatureRegression(output_dim=output_dim,
                                                   feature_size=fr_feature_size,
                                                   kernel_sizes=fr_kernel_sizes,
                                                   channels=fr_channels,
                                                   batch_normalization=batch_normalization)

    def forward(self, tnf_batch):
        feature_A = self.FeatureExtraction.forward(tnf_batch['source image'])
        feature_B = self.FeatureExtraction.forward(tnf_batch['target image'])

        correlation = self.FeatureCorrelation.forward(feature_A, feature_B)

        theta = self.FeatureRegression.forward(correlation)

        if self.return_correlation:
            return (theta, correlation)
        else:
            return theta