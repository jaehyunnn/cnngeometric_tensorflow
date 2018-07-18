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

def save_checkpoint(state, is_best, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir != '' and not exists(model_dir):
        makedirs(model_dir)
    saver = tf.train.Saver()
    ckpt_path = saver.save(state, file)
    if is_best:
        shutil.copyfile(file, join(model_dir, 'best_' + model_fn))


def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
