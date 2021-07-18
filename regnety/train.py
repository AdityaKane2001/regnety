import tensorflow as tf
import argparse
import os
import json
import regnety

from datetime import datetime
from regnety.regnety.models.model import RegNetY
from regnety.regnety.dataset.imagenet import ImageNet
from regnety.regnety.utils import train_utils as tutil
from regnety.regnety.config.config import (
    get_train_config,
    get_custom_train_config,
    get_preprocessing_config,
    ALLOWED_FLOPS
)


parser = argparse.ArgumentParser(description="Train RegNetY")
parser.add_argument("-f", "--flops", type=str, help="FLOP variant of RegNetY")
parser.add_argument("-taddr","--tpu_address", type=str, help="Network address of TPU clsuter",default=None)
parser.add_argument("-tfp","--tfrecs_path_pattern",type=str,help="GCS bucket path pattern for tfrecords")
parser.add_argument("-trial", "--trial_run", action='store_true')

args = parser.parse_args()
flops = args.flops
tpu_address = args.tpu_address
tfrecs_filepath = tf.io.gfile.glob(args.tfrecs_path_pattern)
trial = args.trial_run

if "mf" not in flops:
    flops += "mf"

if flops not in ALLOWED_FLOPS:
    raise ValueError("Flops must be one of %s. Received: %s" % (ALLOWED_FLOPS, 
        flops.rstrip('mf')))

cluster_resolver, strategy = tutil.connect_to_tpu(tpu_address)

if trial:
    train_cfg = get_custom_train_config(
        optimizer="sgd",
        base_lr=0.1 * strategy.num_replicas_in_sync,
        warmup_epochs=5,
        warmup_factor=0.1,
        total_epochs=100,
        weight_decay=5e-4,
        momentum=0.9,
        lr_schedule="half_cos",
        log_dir="gs://adityakane-train/logs",
        model_dir="gs://adityakane-train/models"
    )
else:
    train_cfg = get_train_config()

prep_cfg = get_preprocessing_config( 
    tfrecs_filepath=tfrecs_filepath,
    batch_size=1024,
    image_size=512,
    crop_size=224,
    resize_to_size=320,
    augment_fn="default",
    num_classes=1000,
    percent_valid=1,
    cache_dir="gs://adityakane-train/cache/",
    color_jitter=False,
    scale_to_unit=True
) 


print('Training options detected:', train_cfg)
print('Preprocessing options detected:', prep_cfg)

with strategy.scope():
    model = tutil.make_model(flops, train_cfg)

train_ds, val_ds = ImageNet(prep_cfg).make_dataset()

trial_callbacks = [
    tf.keras.callbacks.LearningRateScheduler(tutil.get_train_schedule(train_cfg))
]

callbacks = trial_callbacks if trial else tutil.get_callbacks(train_cfg)  

history = model.fit(
    train_ds,
   	epochs=train_cfg.total_epochs,
   	validation_data=val_ds,
   	callbacks=callbacks
)

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%Hh%Mm")

with tf.io.gfile.GFile(os.path.join(train_cfg.log_dir, 'history_%s.json' % date_time), 'a+') as f:
    json.dumps(str(history.history), f)
