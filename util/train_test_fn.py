from __future__ import print_function, division
import numpy as np
from random import choice
import timeit
import matplotlib.pyplot as plt

def train(epoch,cost,optimizer,dataset,pair_generation_tnf, sess, batch_size, source_train, target_train, theta_GT):
    epoch_start = timeit.default_timer()
    avg_cost_train = 0
    total_batch = int(len(dataset) / batch_size)
    for batch_idx in range(1, total_batch + 1):
        # Create mini-batch
        batch = choice(dataset)
        batch['image'] = np.expand_dims(batch['image'], 0)
        batch['theta'] = np.expand_dims(batch['theta'], 0)

        for j in range(batch_size - 1):
            temp = choice(dataset)
            temp['image'] = np.expand_dims(temp['image'], 0)
            temp['theta'] = np.expand_dims(temp['theta'], 0)
            batch['image'] = np.concatenate((batch['image'], temp['image']), 0)
            batch['theta'] = np.concatenate((batch['theta'], temp['theta']), 0)

        data_batch = pair_generation_tnf(batch)
        source_batch = data_batch['source_image']
        target_batch = data_batch['target_image']
        theta_batch = data_batch['theta_GT']

        batch_xs_source, batch_xs_target, batch_ys = source_batch, target_batch, theta_batch

        """
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(source_batch[0])
        axs[0].set_title('source')
        axs[1].imshow(target_batch[0])
        axs[1].set_title('target')
        plt.show()
        """

        # Feed forward
        feed_dict = {source_train: batch_xs_source, target_train: batch_xs_target, theta_GT: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost_train += c / total_batch
        if (batch_idx) % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(epoch, batch_idx, total_batch,
                                                                             100. * batch_idx / total_batch, c))
    epoch_end = timeit.default_timer()
    t = epoch_end - epoch_start
    print('Train set: Average loss= {:.4f}'.format(avg_cost_train),
          '\tTime per epoch: %dm %ds' % ((t / 60), (t % 60)))
    return avg_cost_train