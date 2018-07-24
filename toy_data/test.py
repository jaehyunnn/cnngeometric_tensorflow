import tensorflow as tf
import tensorflow_hub as hub

# Prepare an image tensor.
value = tf.read_file('airport_1.jpeg')
image = tf.image.decode_jpeg(value, channels=3)
image = tf.image.convert_image_dtype(image, tf.float32)

# Instantiate the DELF module.
delf_module = hub.Module("https://tfhub.dev/google/delf/1")

delf_inputs = {
  # An image tensor with dtype float32 and shape [height, width, 3], where
  # height and width are positive integers:
  'image': image,
  # Scaling factors for building the image pyramid as described in the paper:
  'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
  # Image features whose attention score exceeds this threshold will be
  # returned:
  'score_threshold': 100.0,
  # The maximum number of features that should be returned:
  'max_feature_num': 1000,
}

# Apply the DELF module to the inputs to get the outputs.
delf_outputs = delf_module(delf_inputs, as_dict=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(delf_outputs))