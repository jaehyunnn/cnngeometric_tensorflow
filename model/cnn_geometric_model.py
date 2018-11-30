import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v2 as resnet
from tensorflow.contrib.slim.python.slim.nets import vgg
import tensorflow.contrib.slim as slim
from model.nets import resnet101
from model.nets import vgg16

class FeatureExtraction:
    def __init__(self, trainable=False, feature_extraction_cnn='resnet101'):
        if feature_extraction_cnn == 'vgg':
            self.model = vgg16

        if feature_extraction_cnn == 'resnet101':
            self.model = resnet101

    def __call__(self, image_batch):
        if self.model == vgg16:
            with slim.arg_scope(vgg.vgg_arg_scope()):
                features,_ = vgg16(inputs=image_batch)

        if self.model == resnet101:
            with slim.arg_scope(resnet.resnet_arg_scope()):
                features,_ = resnet101(inputs=image_batch,num_classes=None)

        return features

class FeatureL2Norm:
    def __init__(self):
        pass
    def __call__(self, features):
        features = tf.nn.l2_normalize(features, epsilon=(1e-6))
        return features

class FeatureCorrelation:
    def __init__(self):
        pass

    def __call__(self, feature_A, feature_B):
        """
        :param b: batch_size
        :param c: channel(amount of filters)
        :param h: height
        :param w: width
        :return:
        """
        b,h,w,c = feature_A.get_shape().as_list()


        feature_A = tf.transpose(feature_A, [0,2,1,3])
        feature_A = tf.reshape(feature_A, [-1,h*w,c])

        feature_B = tf.reshape(feature_B, [-1,h*w,c])
        feature_B = tf.transpose(feature_B, [0,2,1])

        feature_mul = tf.matmul(feature_A, feature_B)
        correlation_tensor = tf.reshape(feature_mul, [-1,h,w,h*w])

        return correlation_tensor

class FeatureRegression:
    def __init__(self, output_dim=6, batch_normalization=True, kernel_sizes=[7,5], channels=[128,64], feature_size=15):
        self.output_dim = output_dim
        self.batch_normalization = batch_normalization
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.feature_size = feature_size

    def __call__(self, x):
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

class CNNGeometric:
    def __init__(self, output_dim=6,
                 feature_extraction_cnn='resnet101',
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
                                                   feature_extraction_cnn=feature_extraction_cnn)
        self.FeatureCorrelation = FeatureCorrelation()
        self.FeatureRegression = FeatureRegression(output_dim=output_dim,
                                                   feature_size=fr_feature_size,
                                                   kernel_sizes=fr_kernel_sizes,
                                                   channels=fr_channels,
                                                   batch_normalization=batch_normalization)
        self.FeatureL2Norm = FeatureL2Norm()

    def __call__(self, tnf_batch):
        # do feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])
        # normalize
        if self.normalize_features:
            feature_A = self.FeatureL2Norm(feature_A)
            feature_B = self.FeatureL2Norm(feature_B)
        # do feature correlation
        correlation = self.FeatureCorrelation(feature_A, feature_B)
        # normalize
        if self.normalize_matches:
            correlation = self.FeatureL2Norm(tf.nn.relu(correlation))
        # do regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)

        return theta