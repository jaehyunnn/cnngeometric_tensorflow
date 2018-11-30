import shutil
import tensorflow as tf
from os import makedirs, remove
from os.path import exists, join, basename, dirname


class BatchTensorToVars:
    """Convert tensors in dict batch to vars
    """
    def __init__(self):
        pass
    def __call__(self, batch):
        batch_var = {}
        for key, value in batch.items():
            batch_var[key] = tf.Variable(value, trainable=False)

        return batch_var

def save_checkpoint(sess, saver, file):
    save_path = saver.save(sess, file)
    print("Checkpoint saved in path: %s" % save_path)
    return save_path

def reload_checkpoint(sess, saver, file):
    saver.restore(sess, file)
    print('Checkpoint reloaded from -- [' + file + ']')

def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
