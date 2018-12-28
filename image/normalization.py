import numpy as np

class NormalizeImageDict(object):
    def __init__(self, image_keys, normalizeRange=True):
        self.image_keys = image_keys
        self.normalizeRange = normalizeRange

    def __call__(self, sample):
        for key in self.image_keys:
            if self.normalizeRange:
                sample[key] /= 255.0

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = np.array(mean, dtype='float32')
        mean = np.expand_dims(np.expand_dims(mean, 0), 0)
        std = np.array(std, dtype='float32')
        std = np.expand_dims(np.expand_dims(std, 0), 0)

        tile_arg = np.divide(np.array(sample[key].shape), np.array(mean.shape)).astype('int32')
        sample[key] = np.divide(np.subtract(sample[key], np.tile(mean, tile_arg)), np.tile(std, tile_arg))

        return sample

def normalize_image(image, forward=True, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    im_size = image.shape
    mean = np.array(mean, dtype='float32')
    mean = np.expand_dims(np.expand_dims(mean, 0), 0)
    std = np.array(std, dtype='float32')
    std = np.expand_dims(np.expand_dims(std, 0), 0)

    if isinstance(image, np.ndarray):
        mean = np.array(mean)
        std = np.array(std)
    # Normalize
    if forward:
        if len(im_size) == 3:
            tile_arg = np.divide(np.array(im_size), np.array(mean.shape)).astype('int32')
            result = np.divide(np.subtract(image, np.tile(mean, tile_arg)), np.tile(std, tile_arg))
        elif len(im_size) == 4:
            mean = np.expand_dims(mean,0)
            std = np.expand_dims(std,0)
            tile_arg = np.divide(np.array(im_size),np.array(mean.shape)).astype('int32')
            result = np.divide(np.subtract(image,np.tile(mean, tile_arg)),
                               np.tile(std, tile_arg))
    # Un-normalize
    else:
        if len(im_size) == 3:
            tile_arg = np.divide(np.array(im_size), np.array(mean.shape)).astype('int32')
            result = np.add(np.multiply(image, np.tile(std, tile_arg)),
                            np.tile(mean, tile_arg))
        elif len(im_size) == 4:
            mean = np.expand_dims(mean, 0)
            std = np.expand_dims(std, 0)
            tile_arg = np.divide(np.array(im_size), np.array(mean.shape)).astype('int32')
            result = np.add(np.multiply(image,np.tile(std, tile_arg)),
                            np.tile(mean, tile_arg))

    return np.float32(result)