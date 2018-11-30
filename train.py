from __future__ import print_function, division
import argparse
import os
from os.path import exists, join, basename
import tensorflow as tf
from model.cnn_geometric_model import CNNGeometric
from model.loss import TransformedGridLoss
from data.synth_dataset import SynthDataset
from data.download_datasets import download_pascal
from geotnf.transformation import SynthPairTnf
from image.normalization import NormalizeImageDict
from util.train_test_fn import train
from util.tf_util import save_checkpoint, str_to_bool


"""

Script to train the model as presented in the CNNGeometric CVPR'17 paper
using synthetically warped image pairs and strong supervision

"""

print('CNNGeometric training script')

# Argument parsing
parser = argparse.ArgumentParser(description='CNNGeometric TensorFlow implementation')
# Paths
parser.add_argument('--training-dataset', type=str, default='pascal', help='dataset to use for training')
parser.add_argument('--training-tnf-csv', type=str, default='', help='path to training transformation csv folder')
parser.add_argument('--training-image-path', type=str, default='', help='path to folder containing training images')
parser.add_argument('--trained-models-dir', type=str, default='trained_models', help='path to trained models folder')
parser.add_argument('--trained-models-fn', type=str, default='checkpoint_adam', help='trained model filename')
# Optimization parameters
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
parser.add_argument('--num-epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=16, help='training batch size')
parser.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
parser.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
# Model parameters
parser.add_argument('--geometric-model', type=str, default='affine',
                    help='geometric model to be regressed at output: affine or tps')
parser.add_argument('--use-mse-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use MSE loss on tnf. parameters')
parser.add_argument('--feature-extraction-cnn', type=str, default='vgg', help='Feature extraction architecture: vgg/resnet101')
# Synthetic dataset parameters
parser.add_argument('--random-sample', type=str_to_bool, nargs='?', const=True, default=False,
                    help='sample random transformations')

args = parser.parse_args()

# Seed
tf.set_random_seed(args.seed)

# Download dataset if needed and set paths
if args.training_dataset == 'pascal':
    if args.training_image_path == '':
        download_pascal('datasets/pascal-voc11/')
        args.training_image_path = 'datasets/pascal-voc11/'
    if args.training_tnf_csv == '' and args.geometric_model == 'affine':
        args.training_tnf_csv = 'training_data/pascal-synth-aff'
    elif args.training_tnf_csv == '' and args.geometric_model == 'tps':
        args.training_tnf_csv = 'training_data/pascal-synth-tps'

# CNN model and loss
print('Creating CNN model...')

model = CNNGeometric(feature_extraction_cnn=args.feature_extraction_cnn)

if args.use_mse_loss:
    print('Using MSE loss...')
    loss = tf.losses
else:
    print('Using grid loss...')
    loss = TransformedGridLoss(geometric_model=args.geometric_model)

# Dataset and dataloader
dataset = SynthDataset(geometric_model=args.geometric_model,
                       csv_file=os.path.join(args.training_tnf_csv, 'train.csv'),
                       training_image_path=args.training_image_path,
                       transform=NormalizeImageDict(['image']),
                       random_sample=args.random_sample)

dataset_test = SynthDataset(geometric_model=args.geometric_model,
                            csv_file=os.path.join(args.training_tnf_csv, 'test.csv'),
                            training_image_path=args.training_image_path,
                            transform=NormalizeImageDict(['image']),
                            random_sample=args.random_sample)

pair_generation_tnf = SynthPairTnf(geometric_model=args.geometric_model, output_size=(240, 240))


source_train = tf.placeholder(tf.float32, [None, 240, 240, 3])
target_train = tf.placeholder(tf.float32, [None, 240, 240, 3])
input_pair_train = {'source_image':source_train, 'target_image':target_train}
theta_GT = tf.placeholder(tf.float32, [None, 2, 3])

# Model operation
theta = model(input_pair_train)

# Optimizer
cost = loss(theta=theta, theta_GT=theta_GT, batch_size=args.batch_size)
optimizer = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(cost)

# Train
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('Starting training...\n')
    print("# ===================================== #")
    print("\t\t......Train config......")
    print("\t\t CNN model: ", args.feature_extraction_cnn)
    print("\t\t Geometric model: ", args.geometric_model)
    print("\t\t Dataset: ", args.training_dataset)
    print()
    print("\t\t Learning rate: ", args.lr)
    print("\t\t Batch size: ", args.batch_size)
    print("\t\t Maximum epoch: ", args.num_epochs)
    print("# ===================================== #\n")

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train(epoch=epoch, cost=cost, optimizer=optimizer, dataset=dataset, pair_generation_tnf=pair_generation_tnf,
                           sess=sess, batch_size=args.batch_size, source_train=source_train, target_train=target_train, theta_GT=theta_GT)

        # Save checkpoint
        if args.use_mse_loss:
            checkpoint_name = join(args.trained_models_dir,
                                   args.trained_models_fn + '_' + args.geometric_model + '_mse_loss_' + args.feature_extraction_cnn + '_' + args.training_dataset + '_epoch_' + str(epoch) + '.ckpt')
        else:
            checkpoint_name = join(args.trained_models_dir,
                                   args.trained_models_fn + '_' + args.geometric_model + '_grid_loss_' + args.feature_extraction_cnn + '_' + args.training_dataset + '_epoch_' + str(epoch) + '.ckpt')
        saver.save(sess, checkpoint_name)
        # save_checkpoint(sess, checkpoint_name)

print('Training Finished!')


# TODO:
#  - Implement 'tps' train code
#  - Implement demo.py
#  - Implement eval_pf.py