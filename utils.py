# lets define some utility functions
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.io import imread

# Image params
# ----------------------------------------------------------------
IMG_WIDTH = 768
IMG_HEIGHT = 768
IMG_CHANNELS = 3
TARGET_WIDTH = 128
TARGET_HEIGHT = 128
image_shape = (768, 768)
# ----------------------------------------------------------------

no_mask = np.zeros(image_shape[0]*image_shape[1], dtype=np.uint8)


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    return rle


def rle_decode(mask_rle, shape=image_shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    if pd.isnull(mask_rle):
        img = no_mask
        return img.reshape(shape).T
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def get_image(image_name):
    img = imread('train_v2/' + image_name)[:, :, :IMG_CHANNELS]
    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT), mode='constant', preserve_range=True)
    return img


def get_mask(code):
    img = rle_decode(code)
    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT, 1), mode='constant', preserve_range=True)
    return img


def get_test_image(image_name):
    img = imread('../input/test_v2/' + image_name)[:, :, :IMG_CHANNELS]
    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT), mode='constant', preserve_range=True)
    return img


def create_test_generator(precess_batch_size, sub_df):
    while True:
        for k, ix in sub_df.groupby(np.arange(sub_df.shape[0]) // precess_batch_size):
            imgs = []
            for index, row in ix.iterrows():
                original_img = get_test_image(row.ImageId) / 255.0
                imgs.append(original_img)

            imgs = np.array(imgs)
            yield imgs

# Function to create generator


def create_image_generator(precess_batch_size, data_df):
    while True:
        for k, group_df in data_df.groupby(np.arange(data_df.shape[0]) // precess_batch_size):
            imgs = []
            labels = []
            for index, row in group_df.iterrows():
                # images
                original_img = get_image(row.ImageId) / 255.0
                # masks
                mask = get_mask(row.EncodedPixels) / 255.0

                imgs.append(original_img)
                labels.append(mask)

            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels
