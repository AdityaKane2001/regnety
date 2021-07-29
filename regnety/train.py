import tensorflow as tf
import argparse
import os
import json
import wandb
import logging
import math

from datetime import datetime
from wandb.keras import WandbCallback
from regnety.regnety.models.model import RegNetY
from regnety.regnety.dataset.imagenet import ImageNet
from regnety.regnety.utils import train_utils as tutil
from regnety.regnety.config.config import (
    get_train_config,
    get_preprocessing_config,
    ALLOWED_FLOPS
)


parser = argparse.ArgumentParser(description="Train RegNetY")
parser.add_argument("-f", "--flops", type=str, help="FLOP variant of RegNetY")
parser.add_argument("-taddr", "--tpu_address", type=str,
                    help="Network address of TPU clsuter", default=None)
parser.add_argument("-tfp", "--tfrecs_path_pattern", type=str,
                    help="GCS bucket path pattern for tfrecords")
parser.add_argument("-trial", "--trial_run", action="store_true")

args = parser.parse_args()
flops = args.flops
tpu_address = args.tpu_address
tfrecs_filepath = tf.io.gfile.glob(args.tfrecs_path_pattern)

tfrecs_filepath.sort()
one_percent = math.ceil(len(tfrecs_filepath) / 100)
train_tfrecs_filepath = tf.io.gfile.glob("gs://adityakane-imagenet-tfrecs/train_*.tfrecord")
val_tfrecs_filepath = tf.io.gfile.glob("gs://adityakane-imagenet-tfrecs/valid_*.tfrecord")
trial = args.trial_run

logging.basicConfig(format="%(asctime)s %(levelname)s : %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO)

if "mf" not in flops:
    flops += "mf"

if flops not in ALLOWED_FLOPS:
    raise ValueError("Flops must be one of %s. Received: %s" % (ALLOWED_FLOPS,
                                                                flops.rstrip("mf")))

cluster_resolver, strategy = tutil.connect_to_tpu(tpu_address)

train_cfg = get_train_config(
    optimizer="sgd",
    base_lr=0.1 * strategy.num_replicas_in_sync,
    warmup_epochs=5,
    warmup_factor=0.1,
    total_epochs=100,
    weight_decay=5e-5,
    momentum=0.9,
    lr_schedule="half_cos",
    log_dir="gs://adityakane-train/logs",
    model_dir="gs://adityakane-train/models",
)


train_prep_cfg = get_preprocessing_config(
    tfrecs_filepath=train_tfrecs_filepath,
    augment_fn="default",
    mixup=True
)

val_prep_cfg = get_preprocessing_config(
    tfrecs_filepath=val_tfrecs_filepath,
    augment_fn="val",
    mixup=False
)


logging.info(f"Training options detected: {train_cfg}")
logging.info("Preprocessing options detected.")
logging.info(
    f"Training on TFRecords: {train_prep_cfg.tfrecs_filepath[0]} to {train_prep_cfg.tfrecs_filepath[-1]}")
logging.info(
    f"Validating on TFRecords: {val_prep_cfg.tfrecs_filepath[0]} to {val_prep_cfg.tfrecs_filepath[-1]}")

with strategy.scope():
    model = tutil.make_model(flops, train_cfg)

train_ds = ImageNet(train_prep_cfg).make_dataset()
val_ds = ImageNet(val_prep_cfg).make_dataset()

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%Hh%Mm")

wandb.init(entity="compyle", project="regnety",
           job_type="train", name="Final_" + date_time + "_" + flops.upper())

trial_callbacks = [
    tf.keras.callbacks.LearningRateScheduler(
        tutil.get_train_schedule(train_cfg)),
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(train_cfg.log_dir, str(date_time)), histogram_freq=1),  # profile_batch="0,1023"
    WandbCallback()
]

callbacks = trial_callbacks if trial else tutil.get_callbacks(
    train_cfg, date_time)

history = model.fit(
    train_ds,
   	epochs=train_cfg.total_epochs,
   	validation_data=val_ds,
   	callbacks=callbacks
)

with tf.io.gfile.GFile(os.path.join(train_cfg.log_dir, "history_%s.json" % date_time), "a+") as f:
   json.dump(str(history.history), f)
