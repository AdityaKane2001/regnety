"""Script to evaluate trained RegNetY models."""

import tensorflow as tf
import argparse
import logging

from regnety.dataset.imagenet import ImageNet
from regnety.utils import train_utils as tutil
from regnety.config.config import (
    get_train_config,
    get_preprocessing_config,
    ALLOWED_FLOPS,
)


parser = argparse.ArgumentParser(description="Evaluate trained models")
parser.add_argument(
    "-path", "--model_path", type=str, help="Path to saved models directory"
)
parser.add_argument(
    "-tfrecs",
    "--tfrecs_path_pattern",
    type=str,
    help="Path pattern for tfrecords",
)
parser.add_argument("-f", "--flops", type=str, help="FLOP variant of RegNetY")
args = parser.parse_args()

tfrecs_filepath = tf.io.gfile.glob(args.tfrecs_path_pattern)
model_path = args.model_path
flops = args.flops
if "mf" not in flops:
    flops += "mf"

if flops not in ALLOWED_FLOPS:
    raise ValueError(
        "Flops must be one of %s. Received: %s" % (ALLOWED_FLOPS, flops.rstrip("mf"))
    )


logging.basicConfig(
    format="%(asctime)s %(levelname)s : %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


eval_prep_cfg = get_preprocessing_config(
    tfrecs_filepath=tfrecs_filepath, augment_fn="val"
)
logging.info("Preprocessing options detected.")
logging.info(
    f"Validating on TFRecords: {eval_prep_cfg.tfrecs_filepath[0]} to {eval_prep_cfg.tfrecs_filepath[-1]}"
)

cluster_resolver, strategy = tutil.connect_to_tpu()

# The values of the attributes need not be same as training
eval_cfg = get_train_config(
    optimizer="adamw",
    base_lr=0.001 * strategy.num_replicas_in_sync,
    warmup_epochs=5,
    warmup_factor=0.1,
    total_epochs=100,
    weight_decay=5e-5,
    momentum=0.9,
    lr_schedule="half_cos",
    log_dir="",
    model_dir="",
)


with strategy.scope():
    model = tutil.make_model(flops, eval_cfg)

model.load_weights(model_path)

eval_ds = ImageNet(eval_prep_cfg).make_dataset()

logging.info("Model and dataset initialized")

eval_dict = model.evaluate(eval_ds, return_dict=True)

logging.info(f"Model evaluation: {eval_dict}")
