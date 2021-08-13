import tensorflow as tf
import argparse
import os
import json
import wandb
import logging
import time
import math

from datetime import datetime
from wandb.keras import WandbCallback
from regnety.models.model import RegNetY
from regnety.dataset.imagenet import ImageNet
from regnety.utils import train_utils as tutil
from regnety.config.config import (
    get_train_config,
    get_preprocessing_config,
    ALLOWED_FLOPS
)


parser = argparse.ArgumentParser(description="Benchmark RegNetY")
parser.add_argument("-f", "--flops", type=str, help="FLOP variant of RegNetY")
parser.add_argument("-m", "--model_location", help="SavedModel directory path")
parser.add_argument("-tfrec", "--tfrecords_bucket_path", help="Path to TFRecords bucket. eg. gs://abcd.")

args = parser.parse_args()
flops = args.flops
model_location = args.model_location
imgnet_location = args.tfrecords_bucket_path

BATCH_SIZE = 1024

logging.basicConfig(format="%(asctime)s %(levelname)s : %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO)

logging.info("Benchmarking on ImageNet validation dataset")

val_tfrecs_filepath = tf.io.gfile.glob(imgnet_location + "/valid_*.tfrecord")

val_prep_cfg = get_preprocessing_config(
    batch_size=BATCH_SIZE,
    tfrecs_filepath=val_tfrecs_filepath,
    augment_fn="val",
    mixup=False
)

val_ds = ImageNet(val_prep_cfg, no_aug=True).make_dataset()
val_ds = val_ds.repeat()

cluster_resolver, strategy = tutil.connect_to_tpu()

with strategy.scope():
    model = tf.keras.models.load_model(model_location)

#warmup device, iterator
model.predict(val_ds, steps=10)
logging.info("Warmed up.")

NUM_IMAGES = BATCH_SIZE * 100

logging.info("Running benchmark.")
start = time.time()
model.predict(val_ds, steps=100)
end = time.time()
time_taken = end - start

logging.info(f"Inference of {NUM_IMAGES} images took {time_taken} seconds. images/second = {NUM_IMAGES / time_taken}")

