import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import variable_scope


def resnet101(inputs,num_classes=None,
              is_training=True,
              global_pool=False,
              output_stride=None,
              spatial_squeeze=False,
              reuse=tf.AUTO_REUSE,
              scope='resnet_v2_101'):
  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2.resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2.resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2.resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
  ]
  return resnet_v2.resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True,
                   reuse=reuse, scope=scope)

def vgg16(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          reuse=tf.AUTO_REUSE,
          scope='vgg_16'):
  with variable_scope.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope([layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],outputs_collections=end_points_collection):
      net = layers_lib.repeat(
          inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
      net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
      net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')

      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      return net, end_points