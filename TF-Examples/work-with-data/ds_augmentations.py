from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

from write_read_records import feature_description
from augmentation_model import seq

BATCH_SIZE = 4


def tf_random_augmentations(image, label):
    def augment_image(image):
        return seq(images=[image.numpy()])

    [image, ] = tf.py_function(augment_image, [image], [tf.uint8])
    return image, label


def _parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.cast(tf.io.decode_jpeg(example['image'], channels=3), tf.float32)
    name = example['name']
    return image, name


def normalize_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    return image, label


if __name__ == '__main__':
    rec_path = Path("./records")
    records = [str(file) for file in rec_path.glob("*")]

    ds = tf.data.TFRecordDataset(records)
    ds = ds.map(_parse_function).map(tf_random_augmentations).map(normalize_image).batch(BATCH_SIZE)

    for images, names in ds:
        for image, name in zip(images, names):
            image = image.numpy()
            name = name.numpy().decode("utf-8")
            plt.title(name)
            plt.imshow(image)
            plt.show()
