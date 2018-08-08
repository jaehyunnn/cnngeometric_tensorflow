import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.python.util.all_util import make_all

model = resnet_v2
def resnet_v2_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=False,
                  output_stride=None,
                  spatial_squeeze=False,
                  reuse=tf.AUTO_REUSE,
                  scope='resnet_v2_101'):
  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      model.resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      model.resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      model.resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
  ]
  return model.resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True,
                   reuse=reuse, scope=scope)