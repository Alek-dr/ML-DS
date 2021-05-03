from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from write_read_records import extract_name

HEIGHT = 370
WIDTH = 278


def encode_single_sample(img_path, encoded_label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [HEIGHT, WIDTH])
    return img, encoded_label


def get_dataset(ds_path: Path, batch_size: int):
    paths = [path for path in ds_path.glob("*.jpg")]
    images = [str(file) for file in paths]
    labels = [extract_name(file) for file in paths]
    assert len(images) == len(labels)
    images = np.array(images)
    labels = np.array(labels)

    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    ds = ds.map(encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
        batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


if __name__ == '__main__':
    ds_path = Path("./data")
    ds = get_dataset(ds_path, batch_size=2)
    for images, names in ds:
        for image, name in zip(images, names):
            image = image.numpy()
            name = name.numpy().decode("utf-8")
            plt.title(name)
            plt.imshow(image)
            plt.show()
