from __future__ import print_function, division
import argparse
import os
from os.path import exists, join, basename
import tensorflow as tf
#from torch.utils.data import Dataset, DataLoader
from model.cnn_geometric_model_complete import CNNGeometric
from model.loss_complete import TransformedGridLoss
from data.synth_dataset_complete import SynthDataset
from data.download_datasets_complete import download_pascal
from geotnf.transformation_complete import SynthPairTnf
from image.normalization_complete import NormalizeImageDict
from util.train_test_fn import train, test
from util.tf_util_complete import save_checkpoint, str_to_bool
from random import choice

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
parser.add_argument('--use-mse-loss', type=str_to_bool, nargs='?', const=True, default=False,
                    help='Use MSE loss on tnf. parameters')
parser.add_argument('--feature-extraction-cnn', type=str, default='resnet_v2',
                    help='Feature extraction architecture: vgg/resnet_v2')
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
"""
dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4) """

dataset_test = SynthDataset(geometric_model=args.geometric_model,
                            csv_file=os.path.join(args.training_tnf_csv, 'test.csv'),
                            training_image_path=args.training_image_path,
                            transform=NormalizeImageDict(['image']),
                            random_sample=args.random_sample)
"""
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                             shuffle=True, num_workers=4) """


pair_generation_tnf = SynthPairTnf(geometric_model=args.geometric_model, output_size=(240, 240))

# Parameter
EPOCHS = 100
LEARNING_RATE = 0.01
BATCH_SIZE = 1

source_train = tf.placeholder(tf.float32, [None, 240, 240, 3])
target_train = tf.placeholder(tf.float32, [None, 240, 240, 3])
input_pair_train = {'source_image':source_train, 'target_image':target_train}

y_train = tf.placeholder(tf.float32, [None, 2, 3])

# Model operation
y_hat = model(input_pair_train)

# Optimizer
cost = loss(theta=y_hat, theta_GT=y_train)
optimizer = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(cost)

# Train
if args.use_mse_loss:
    checkpoint_name = os.path.join(args.trained_models_dir,
                                   args.trained_models_fn + '_' + args.geometric_model + '_mse_loss' + args.feature_extraction_cnn + '.pth.tar')
else:
    checkpoint_name = os.path.join(args.trained_models_dir,
                                   args.trained_models_fn + '_' + args.geometric_model + '_grid_loss' + args.feature_extraction_cnn + '.pth.tar')

best_test_loss = float("inf")

# Make session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Iteration section
print('Learning Started!')
for epoch in range(1, args.num_epochs + 1):
    #train_loss = train(epoch, model, loss, optimizer, dataloader, pair_generation_tnf, log_interval=100)
    #test_loss = test(model, loss, dataloader_test, pair_generation_tnf)

    avg_cost_train = 0
    total_batch = int(len(dataset) / BATCH_SIZE)

    for i in range(total_batch):
        rnd_choice = pair_generation_tnf(choice(dataset))
        batch_xs_source, batch_xs_target, batch_ys = rnd_choice['source_image'], rnd_choice['target_image'], rnd_choice['theta_GT']
        batch_xs_source = batch_xs_source.eval(session=sess)
        batch_xs_target = batch_xs_target.eval(session=sess)
        batch_ys = batch_ys.eval(session=sess)

        feed_dict = {source_train: batch_xs_source, target_train: batch_xs_target, y_train: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost_train += c / total_batch
        print('batch_num: ', '%04d' % (i + 1), 'cost= ', '{:.9f}'.format(c))
    print('Epoch: ', '%04d' % (epoch + 1), 'cost= ', '{:.9f}'.format(avg_cost_train))

"""
TODO : checkpoint 저장 모듈 구현 
    # remember best loss
    is_best = test_loss < best_test_loss
    best_test_loss = min(test_loss, best_test_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'args': args,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint_name)
"""
print('Learning Finished!')
