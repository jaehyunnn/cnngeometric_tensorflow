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
        # batch_time_start = timeit.default_timer()
        batch = dict()
        images = []
        thetas = []
        for j in range(batch_size):
            temp = choice(dataset)
            images.append(temp['image'])
            thetas.append(temp['theta'])
        batch['image'] = np.array(images)
        batch['theta'] = np.array(thetas)

        # batch_time_end=timeit.default_timer()
        # print("batch gen time:",batch_time_end-batch_time_start)
        data_batch = pair_generation_tnf(batch)
        # pair_time_end = timeit.default_timer()
        # print("pair gen time:", pair_time_end - batch_time_end)
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
                                                                             100. * batch_idx / total_batch, c), flush=True)
    epoch_end = timeit.default_timer()
    t = epoch_end - epoch_start
    print('Train set: Average loss= {:.4f}'.format(avg_cost_train),
          '\tTime per epoch: %dm %ds' % ((t / 60), (t % 60)))

    return avg_cost_train