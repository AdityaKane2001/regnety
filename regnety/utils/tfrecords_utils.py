"""Script to make and save TFRecords from ImageNet files"""

import tensorflow as tf
import os
import random
import json
import argparse
import math
from .image_utils import *


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = (
            value.numpy()
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# make_tfrecs -> for(_make_single_tfrecord)-> make_dataset -> _make_example
def _get_synset_labels(filepath):
    """
    Gets synsets from json file in a dict 
    Args:
        filepath: json file path
    
    Returns:
        Dict having the following structure:
        {str id : (int synset_ID, str label_name )}
    
    """

    with open(filepath, "r") as f:
        raw_labels_dict = json.load(f)
    labels_dict = dict()

    for i in raw_labels_dict:
        labels_dict[raw_labels_dict[i]["id"]] = (
            int(i),
            raw_labels_dict[i]["label"],
        )
    return labels_dict


def _get_files(data_dir, synset_filepath, shuffle = True):
    """
    Returns lists of all files, their integer labels and their synsets
    Args:
        data_dir: directory containing ImageNet-style directory structure 
            (synsets ID as directory names, images inside)
        synset_filepath: path to synsets json file
        shuffle: True if data needs to be shuffled    
    
    Returns:
        all_images: paths to all image files
        all_labels: integer labels corresponding to images in all_images list
        all_synsets: synset strings corresponding to images in all_images list
    """
    all_images = tf.io.gfile.glob(os.path.join(data_dir, "*", "*.JPEG"))
    all_synsets = [os.path.basename(os.path.dirname(f)) for f in all_images]

    print(len(all_images))
    all_indexes = list(range(len(all_images)))
    if shuffle:
        random.shuffle(all_indexes)

    all_images = [all_images[i] for i in all_indexes]
    all_synsets = [all_synsets[i][1:] + "-n" for i in all_indexes]

    labels_dict = _get_synset_labels(synset_filepath)

    all_labels_int = [labels_dict[i][0] for i in all_synsets]

    return all_images, all_labels_int, all_synsets


def _make_image(filepath):
    """
    Reads an image and returns its raw byte string. Converts all images to JPEG 
    RGB. 

    Args:
        filepath: path to .JPEG image

    Returns:
        A byte string of the image with JPEG RGB format. 
    """

    image_str = tf.io.read_file(filepath)
    

    if is_png(filepath):
        image_str = png_to_jpeg(image_str)

    if is_cmyk(filepath):
        image_str = cmyk_to_rgb(image_str)

    image_tensor = tf.io.decode_jpeg(image_str)
    height, width = image_tensor.shape[0], image_tensor.shape[1] 

    if not is_rgb(image_tensor):
        image_tensor = tf.image.grayscale_to_rgb(image_tensor)

    image_str = tf.io.encode_jpeg(image_tensor)

    assert len(image_tensor.shape) == 3

    return image_str, height, width


def _make_example(image_str, height, width, filepath, label, synset):
    """
    Makes a single example from arguments

    Args:
        image_str: bytes string of image in JPEG RGB format
        filepath: path to image
        height: height of image in pixels
        width: width of image in pixels
        label: integer denoting label
        synset: synset string corresponding to image
    
    Returns:
        A tf.train.Example having aforementioned attributes
    """

    try:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image": _bytes_feature(image_str),
                    "height": _int64_feature(height),
                    "width": _int64_feature(width),
                    "filename": _bytes_feature(
                        bytes(os.path.basename(filepath)).encode("utf8") 
                    ),
                    "label": _int64_feature(label),
                    "synset": _bytes_feature(bytes(synset).encode("utf8")),
                }
            )
        )
    except TypeError:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image": _bytes_feature(image_str),
                    "height": _int64_feature(height),
                    "width": _int64_feature(width),
                    "filename": _bytes_feature(
                        bytes(os.path.basename(filepath), encoding="utf8")
                    ),
                    "label": _int64_feature(label),
                    "synset": _bytes_feature(bytes(synset, encoding="utf8")),
                }
            )
        )
    return example



def _make_single_tfrecord(
    chunk_files, chunk_synsets, chunk_labels, output_filepath
):

    """
    Creates a single TFRecord file having batch_size examples.

    Args: 
        chunk_files: list of filepaths to images
        chunk_synsets: list of synsets corresponding to images in chunk_files
        chunk_labels: list of integer labels corresponding to images in 
            chunk_files
        output_filepath: Output tfrecord file

    Returns None 
    """

    with tf.io.TFRecordWriter(output_filepath) as writer:
        for i in range(len(chunk_files)):
            image_str, height, width = _make_image(chunk_files[i])
            label = chunk_labels[i]
            synset = chunk_synsets[i]

            example = _make_example(
                image_str, height, width, chunk_files[i], label, synset
            )

            writer.write(example.SerializeToString())
    writer.close()


def make_tfrecs(
    dataset_base_dir=None,  # example: home/imagenet/train
    output_dir=None,  # example: home/imagenet_tfrecs
    file_prefix=None,  # example: file_prefix = 'train' makes all files look like: train_0000_of_num_shards
    synset_filepath=None,
    batch_size=1024,
    logging_frequency=1
):
    """
    Only public function of the module. Makes TFReocrds and stores them in 
    output_dir. Each TFRecord except last one has exactly one batch of data. 

    Args:
        dataset_base_dir: directory containing ImageNet-style directory 
            structure (synsets ID as directory names, images inside)
        output_dir: Directory to store TFRecords
        file_prefix: prefix to be added tfrecords files
        synset_filepath: path to synsets json file
        batch_size: batch size of dataset. Each TFRecords, except the last one 
            will contain these many examples.
        logging_frequency: 'Writing shard ..'  will be logged to stdout after 
            these many shards are written.

    Returns None
    """
    if None in (dataset_base_dir, output_dir, file_prefix, synset_filepath):
        raise ValueError('One or more of the arguments is None.')

    images, labels, synsets = _get_files(dataset_base_dir, synset_filepath)

    num_shards = int(math.ceil(len(images) / batch_size))

    for shard in range(num_shards):
        if shard % logging_frequency == 0:
            print("Writing %d of %d shards" % (shard, num_shards))

        chunk_files = images[shard * batch_size : (shard + 1) * batch_size]
        chunk_synsets = synsets[shard * batch_size : (shard + 1) * batch_size]
        chunk_labels = labels[shard * batch_size : (shard + 1) * batch_size]

        output_filepath = os.path.join(
            output_dir, file_prefix + "_%.4d_of_%.4d.tfrecord" % (shard, num_shards)
        )

        _make_single_tfrecord(
            chunk_files, chunk_synsets, chunk_labels, output_filepath
        )
    print('All shards written successfully!')
