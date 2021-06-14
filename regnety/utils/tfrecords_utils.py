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
            value
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# make_tfrecs -> for(make_single_tfrecord)-> make_dataset -> make_example
def get_synset_labels(filepath):

    with open(filepath, "r") as f:
        raw_labels_dict = json.load(f)
    labels_dict = dict()

    for i in raw_labels_dict:
        labels_dict[raw_labels_dict[i]["id"]] = (
            int(i),
            raw_labels_dict[i]["label"],
        )
    return labels_dict


def get_files(data_dir, synset_filepath):
    all_images = tf.io.gfile.glob(os.path.join(data_dir, "*", "*.JPEG"))
    all_synsets = [os.path.basename(os.path.dirname(f)) for f in all_images]

    print(len(all_images))
    all_indexes = list(range(len(all_images)))
    random.shuffle(all_indexes)

    all_images = [all_images[i] for i in all_indexes]
    all_synsets = [all_synsets[i][1:] + "-n" for i in all_indexes]

    labels_dict = get_synset_labels(synset_filepath)

    all_labels_int = [labels_dict[i][0] for i in all_synsets]

    return all_images, all_labels_int, all_synsets


def make_image(filepath):

    image_str = tf.io.read_file(filepath)
    

    if is_png(filepath):
        image_str = png_to_jpeg(image_str)

    if is_cmyk(filepath):
        image_str = cmyk_to_rgb(image_str)

    image_tensor = tf.io.decode_jpeg(image_str)

    if not is_rgb(image_tensor):
        image_tensor = tf.image.grayscale_to_rgb(image_tensor)
        
    image_tensor = tf.cast(tf.image.resize(image_tensor, (512,512)), tf.uint8)

    image_str = tf.io.encode_jpeg(image_tensor)

    assert len(image_tensor.shape) == 3

    return image_str


def make_example(image_str, filepath, label, synset):

    try:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image": _bytes_feature(image_str),
                    "filename": _bytes_feature(
                        os.path.basename(filepath)
                    ),
                    "label": _int64_feature(label),
                    "synset": _bytes_feature(synset),
                }
            )
        )
    except:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image": _bytes_feature(image_str),
                    "filename": _bytes_feature(
                        os.path.basename(filepath)
                    ),
                    "label": _int64_feature(label),
                    "synset": _bytes_feature(synset),
                }
            )
        )
    return example


def make_single_tfrecord(
    chunk_files, chunk_synsets, chunk_labels, output_filepath
):

    with tf.io.TFRecordWriter(output_filepath) as writer:
        for i in range(len(chunk_files)):
            image_str, height, width = make_image(chunk_files[i])
            label = chunk_labels[i]
            synset = chunk_synsets[i]

            example = make_example(
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
):

    """Driver function"""



    images, labels, synsets = get_files(dataset_base_dir, synset_filepath)

    filepaths_ds = tf.data.Dataset.from_tensor_slices([
        (images[i]) for i in range(len(labels))
    ])


    images_ds = filepaths_ds.map(lambda filepath: make_image(filepath))
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    synsets_ds = tf.data.Dataset.from_tensor_slices(synsets)
    filepaths_ds = filepaths_ds.map(lambda filepath: tf.strings.split(filepath, '/')[-1])
    
    ds = tf.data.Dataset.zip((images_ds, filepaths_ds, labels_ds, synsets_ds))  
    ds = ds.batch(batch_size, drop_remainder = False)
    
    num_shards = int(math.ceil(len(images) / batch_size))

    shard = 0
    for (image_str, filename, label, synset) in ds:
        
        output_filepath = os.path.join(
            output_dir, file_prefix + "_%.4d_of_%.4d.tfrecord" % (shard, num_shards)
        )
        print("Writing %d of %d shards"%(shard, num_shards))
        with tf.io.TFRecordWriter(output_filepath) as writer:
            for i in range(len(label)):
                example = make_example(
                    image_str[i].numpy(), filename[i].numpy(), label[i].numpy(), 
                    synset[i].numpy()
            )
                writer.write(example.SerializeToString())
        writer.close()
        shard += 1
    
