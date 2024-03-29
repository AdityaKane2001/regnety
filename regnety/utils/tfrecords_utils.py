"""Script to make and save TFRecords from ImageNet files"""

import apache_beam as beam
import tensorflow as tf
import os
import random
import json
import math

from regnety.utils.image_utils import *
from regnety.utils.beam_utils import *
from typing import Tuple, List
from collections import namedtuple
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# make_tfrecs -> for(_make_single_tfrecord)-> make_dataset -> _make_example


def _get_default_synset_path() -> str:
    self_path = __file__
    path_segments = self_path.split("/")
    regnety_path = "/".join(path_segments[:-2])
    return os.path.join(regnety_path, "config", "imagenet_synset_to_human.json")


def _get_default_validation_labels_path() -> str:
    self_path = __file__
    path_segments = self_path.split("/")
    regnety_path = "/".join(path_segments[:-2])
    return os.path.join(regnety_path, "config", "valid_labels.txt")


def _get_synset_labels(filepath: str) -> dict:
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


def _get_validation_info(
    data_dir: str,
    synset_filepath: str,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Returns lists of all files, their integer labels and their synsets.
    Assumes directory structure like in ImageNet validation images.
    Args:
        data_dir: directory containing ImageNet validation-style directory
            structure (directly images inside)
        synset_filepath: path to synsets json file
        shuffle: True if data needs to be shuffled

    Returns:
        all_images: paths to all image files
        all_labels: integer labels corresponding to images in all_images list
        all_synsets: synset strings corresponding to images in all_images list
    """
    all_images = tf.io.gfile.glob(os.path.join(data_dir, "*.JPEG"))
    all_images.sort()
    with open(_get_default_validation_labels_path(), "r") as f:
        all_lines = f.readlines()
        all_labels_int = list(
            map(lambda line: int(line.split()[1].strip("\n")), all_lines)
        )
    with open(synset_filepath, "r") as f:
        labels_dict = json.load(f)
    print(labels_dict)
    all_synsets = [labels_dict[str(i)]["id"] for i in all_labels_int]

    return all_images, all_labels_int, all_synsets


def _get_files(
    data_dir: str, synset_filepath: str, shuffle: bool = True, val: bool = False
) -> Tuple[List[str], List[int], List[str]]:
    """
    Returns lists of all files, their integer labels and their synsets.
    Assumes directory structure like in ImageNet training images.
    Args:
        data_dir: directory containing ImageNet train-style directory
            structure (synsets ID as directory names, images inside)
            if val = False
        synset_filepath: path to synsets json file
        shuffle: True if data needs to be shuffled

    Returns:
        all_images: paths to all image files
        all_labels: integer labels corresponding to images in all_images list
        all_synsets: synset strings corresponding to images in all_images list
    """

    if val:
        return _get_validation_info(data_dir, synset_filepath)

    labels_dict = _get_synset_labels(synset_filepath)

    all_images = tf.io.gfile.glob(os.path.join(data_dir, "*", "*.JPEG"))

    all_synsets = [os.path.basename(os.path.dirname(f)) for f in all_images]

    all_indexes = list(range(len(all_images)))

    if shuffle and not val:
        random.shuffle(all_indexes)

    all_images = [all_images[i] for i in all_indexes]
    all_synsets = [all_synsets[i][1:] + "-n" for i in all_indexes]

    all_labels_int = [labels_dict[i][0] for i in all_synsets]
    all_synsets = [labels_dict[i][1] for i in all_synsets]

    return all_images, all_labels_int, all_synsets


def _make_image(filepath: str) -> Tuple[str, int, int]:
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

    if not is_rgb(image_tensor):
        image_tensor = tf.image.grayscale_to_rgb(image_tensor)

    image_tensor = tf.cast(tf.image.resize(image_tensor, (512, 512)), tf.uint8)

    image_str = tf.io.encode_jpeg(image_tensor)

    assert len(image_tensor.shape) == 3

    return image_str, 512, 512


def _make_example(
    image_str: bytes, height: int, width: int, filepath: str, label: int, synset: str
) -> tf.train.Example:
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
    chunk_files: List[str],
    chunk_synsets: List[str],
    chunk_labels: List[int],
    output_filepath: str,
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
    dataset_base_dir: str = "",
    output_dir: str = "",
    file_prefix: str = "",
    synset_filepath: str = "",
    batch_size: int = 1024,
    logging_frequency: int = 1,
    shuffle: bool = True,
    val: bool = False,
):
    """
    Makes TFRecords and stores them in output_dir. Each
    TFRecord except last one has exactly one batch of data.

    Args:
        dataset_base_dir: directory containing ImageNet-style directory
            structure (synsets ID as directory names, images inside)
            eg.: home/imagenet/train
        output_dir: Directory to store TFRecords, eg: home/imagenet_tfrecs
        file_prefix: prefix to be added tfrecords files
            eg.: if file_prefix = "train" then
            all files look like: `train_0000_of_<num_shards>`
        synset_filepath: path to synsets json file
        batch_size: batch size of dataset. Each TFRecords, except the last one
            will contain these many examples.
        logging_frequency: "Writing shard .."  will be logged to stdout after
            these many shards are written.
        shuffle: True if dataset needs to be shuffled
    Returns None
    """

    if "" in (dataset_base_dir, output_dir, file_prefix):
        raise ValueError("One or more arguments is not specified.")

    if not os.path.exists(dataset_base_dir):
        raise ValueError("Dataset path does not exist")

    if synset_filepath is "":
        synpath = _get_default_synset_path()
    else:
        synpath = synset_filepath

    print("Synsets filepath: ", synpath)

    images, labels, synsets = _get_files(
        dataset_base_dir, synpath, shuffle=shuffle, val=val
    )

    print("Total images: ", len(images))

    num_shards = int(math.ceil(len(images) / batch_size))

    for shard in range(num_shards):

        if shard % logging_frequency == 0:
            print("Writing %d of %d shards" % (shard, num_shards))

        chunk_files = images[shard * batch_size : (shard + 1) * batch_size]
        chunk_synsets = synsets[shard * batch_size : (shard + 1) * batch_size]
        chunk_labels = labels[shard * batch_size : (shard + 1) * batch_size]

        output_filepath = os.path.join(
            output_dir,
            file_prefix + "_%.4d_of_%.4d.tfrecord" % (shard, num_shards),
        )

        _make_single_tfrecord(chunk_files, chunk_synsets, chunk_labels, output_filepath)
    print("All shards written successfully!")


def make_tfrecs_beam(
    dataset_base_dir: str = "",
    output_dir: str = "",
    file_prefix: str = "",
    synset_filepath: str = "",
    batch_size: int = 1024,
    shuffle: bool = True,
    val: bool = False,
):
    """
    Makes TFRecords and stores them in output_dir using Apache Beam.
    Each TFRecord except last one has exactly one batch of data.

    Args:
        dataset_base_dir: directory containing ImageNet-style directory
            structure (synsets ID as directory names, images inside)
            eg.: home/imagenet/train
        output_dir: Directory to store TFRecords, eg: home/imagenet_tfrecs
        file_prefix: prefix to be added tfrecords files
            eg.: if file_prefix = "train" then
            all files look like: `train_0000_of_<num_shards>`
        synset_filepath: path to synsets json file
        batch_size: batch size of dataset. Each TFRecords, except the last one
            will contain these many examples.
        shuffle: True if dataset needs to be shuffled
    Returns None
    """

    if "" in (dataset_base_dir, output_dir, file_prefix):
        raise ValueError("One or more arguments is not specified.")

    if not os.path.exists(dataset_base_dir):
        raise ValueError("Dataset path does not exist")

    if synset_filepath is "":
        synpath = _get_default_synset_path()
    else:
        synpath = synset_filepath

    images, labels, synsets = _get_files(
        dataset_base_dir, synpath, shuffle=shuffle, val=val
    )

    print("Total images: ", len(images))

    num_shards = int(math.ceil(len(images) / batch_size))

    final_list = [(images[i], labels[i], synsets[i]) for i in range(len(images))]
    args_ = {
        "jobname": "Make TFRecords",
        "runner": "DirectRunner",
        "num_shards": num_shards,
        "prefix": file_prefix,
        "output_dir": output_dir,
        "file_name_suffix": ".tfrecord",
    }

    options = beam.options.pipeline_options.PipelineOptions(**args_)
    args = namedtuple("options", args_.keys())(*args_.values())

    args_ = {
        "jobname": "Make TFRecords",
        "runner": "DirectRunner",
        "num_shards": num_shards,
        "prefix": file_prefix,
        "output_dir": output_dir,
        "file_name_suffix": ".tfrecord",
    }
    options = beam.options.pipeline_options.PipelineOptions(**args_)
    args = namedtuple("options", args_.keys())(*args_.values())

    make_img_dofunc = MakeImageDoFn()
    make_example_dofunc = MakeExampleDoFn()
    batch_examples_transform = beam.transforms.util.BatchElements(
        min_batch_size=batch_size,
        max_batch_size=batch_size,
    )

    write_to_tf_record = beam.io.tfrecordio.WriteToTFRecord(
        file_path_prefix=args.output_dir,
        num_shards=args.num_shards,
        file_name_suffix=args.file_name_suffix,
    )

    with beam.Pipeline(args.runner, options=options) as pipeline:
        _ = (
            pipeline
            | "Make a PCollection" >> beam.Create(final_list)
            | "Batch elements" >> batch_examples_transform
            | "Get image data" >> beam.ParDo(make_img_dofunc)
            | "Serialize data" >> beam.ParDo(make_example_dofunc)
            | "Write to TFRecords files" >> write_to_tf_record
        )
