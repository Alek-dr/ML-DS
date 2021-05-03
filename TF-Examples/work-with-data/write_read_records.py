import contextlib
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from dataset_utils import dataset_util
from dataset_utils import tf_record_creation_util

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'name': tf.io.FixedLenFeature([], tf.string, default_value='')
}


def extract_name(file: Path):
    name = file.stem.split("_")[0]
    return name


def create_tf_example(img_path: Path, width: int, height: int):
    if not img_path.exists():
        return None
    try:
        img = cv2.imread(str(img_path))
    except AttributeError:
        return None
    img = cv2.resize(img, (width, height))
    img = cv2.imencode(img_path.suffix, img)[1].tostring()
    try:
        name = extract_name(img_path)
    except (ValueError, KeyError) as ex:
        print(ex)
        print(img_path)
        return None
    # create example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image': dataset_util.bytes_feature(img),
        'name': dataset_util.bytes_feature(name.encode("utf-8"))
    }))
    return tf_example


def create_records(output_filebase: Path, files: list, num_shards: int, width: int, height: int) -> None:
    with contextlib.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filebase, num_shards)
        index = 0
        for img in tqdm(files, total=len(files), ncols=60):
            tf_example = create_tf_example(img, width, height)
            if tf_example:
                output_shard_index = index % num_shards
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
                index += 1


def _parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.cast(tf.io.decode_jpeg(example['image'], channels=3), tf.float32)
    image = tf.divide(image, 255)
    name = example['name']
    return image, name


def get_dataset(records: list, batch_size: int = 2):
    ds = tf.data.TFRecordDataset(records)
    ds = ds.map(_parse_function).batch(batch_size)
    return ds


def write_records(img_path: Path, out_path: Path):
    files = [img for img in img_path.glob("*.jpg")]
    output_filebase = out_path / "test_dataset.record"
    create_records(output_filebase, files, 1, width=278, height=370)


def read_records(records_path: Path):
    records = [str(file) for file in records_path.glob("*")]
    ds = get_dataset(records)
    for images, names in ds:
        for image, name in zip(images, names):
            image = image.numpy()
            name = name.numpy().decode("utf-8")
            plt.title(name)
            plt.imshow(image)
            plt.show()


if __name__ == '__main__':
    # Write records
    # img_path = Path("./data")
    # out_path = Path("./records")
    # write_records(img_path, out_path)

    # Read records
    rec_path = Path("./records")
    read_records(rec_path)
