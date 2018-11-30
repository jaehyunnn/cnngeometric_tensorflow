import os
import argparse
import tensorflow as tf
from model.cnn_geometric_model import CNNGeometric
from data.pf_dataset import PFDataset
from data.download_datasets import download_PF_willow
from image.normalization import NormalizeImageDict, normalize_image
from geotnf.point_tnf import *
from geotnf.transformation import GeometricTnf
from util.tf_util import reload_checkpoint, BatchTensorToVars
import matplotlib.pyplot as plt
from skimage import io
from random import choice
import numpy as np

print('CNNGeometric PF demo script')

# Argument parsing
parser = argparse.ArgumentParser(description='CNNGeometric TensorFlow implmentation')
# Paths
parser.add_argument('--model-aff', type=str, default='trained_models/checkpoint_adam_affine_grid_loss_vgg_pascal_epoch_8.ckpt', help='Trained affine model filename')
parser.add_argument('--feature-extraction-cnn', type=str, default='resnet101', help='Feature extraction architecture: vgg/resnet101')
parser.add_argument('--pf-path', type=str, default='datasets/PF-dataset', help='Path to PF dataset')

args = parser.parse_args()

# Download dataset if needed
download_PF_willow('datasets/')

# Dataset and dataloader
dataset = PFDataset(csv_file='datasets/PF-dataset/test_pairs_pf.csv',
                    training_image_path=args.pf_path,
                    transform=NormalizeImageDict(['source_image','target_image']))

batchTensorToVars = BatchTensorToVars()

# Instantiate point transformer
pt = PointTnf()

# Instatiate image transformers
affTnf = GeometricTnf(geometric_model='affine')

with tf.Graph().as_default():
    # Create model
    print('Creating CNN model...')
    model_aff = CNNGeometric(feature_extraction_cnn=args.feature_extraction_cnn)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # Load trained weights
    print('Loading trained model weights...')
    saver.restore(sess,args.model_aff)
    # reload_checkpoint(sess, saver, args.model_aff)

    while True:
        # get random batch of size 1
        batch = choice(dataset)
        batch = batchTensorToVars(batch)

        source_im_size = batch['source_im_size']
        target_im_size = batch['target_im_size']

        source_points = batch['source_points']
        target_points = batch['target_points']

        # warp points with estimated transformations
        target_points_norm = PointsToUnitCoords(target_points, target_im_size)

        # Evaluate models
        theta_aff = sess.run()
        warped_image_aff = affTnf(batch['source_image'], np.reshape(theta_aff, [-1, 2, 3]))

        # Un-normalize images and convert to numpy
        source_image = normalize_image(batch['source_image'], forward=False)
        #source_image = source_image.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
        target_image = normalize_image(batch['target_image'], forward=False)
        #target_image = target_image.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
        warped_image_aff = normalize_image(warped_image_aff, forward=False)
        #warped_image_aff = warped_image_aff.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()

        # check if display is available
        exit_val = os.system('python -c "import matplotlib.pyplot as plt;plt.figure()"  > /dev/null 2>&1')
        display_avail = exit_val == 0

        if display_avail:
            N_subplots = 2 + int(do_aff)
            fig, axs = plt.subplots(1, N_subplots)
            axs[0].imshow(source_image)
            axs[0].set_title('src')
            axs[1].imshow(target_image)
            axs[1].set_title('tgt')
            subplot_idx = 2
            if do_aff:
                axs[subplot_idx].imshow(warped_image_aff)
                axs[subplot_idx].set_title('aff')
                subplot_idx += 1

            for i in range(N_subplots):
                axs[i].axis('off')
            print('Showing results. Close figure window to continue')
            plt.show()
        else:
            print('No display found. Writing results to:')
            fn_src = 'source.png'
            print(fn_src)
            io.imsave(fn_src, source_image)
            fn_tgt = 'target.png'
            print(fn_tgt)
            io.imsave(fn_tgt, target_image)
            if do_aff:
                fn_aff = 'result_aff.png'
                print(fn_aff)
                io.imsave(fn_aff, warped_image_aff)

        res = input('Run for another example ([y]/n): ')
        if res == 'n':
            break

# Not re-implented completely