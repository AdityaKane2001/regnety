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
    get_model_config,
    ALLOWED_FLOPS
)


parser = argparse.ArgumentParser(description="Train RegNetY")
parser.add_argument("-f", "--flops", type=str, help="FLOP variant of RegNetY")
parser.add_argument("-taddr","--tpu_address", type=str, help="Network address of TPU clsuter",default=None)
parser.add_argument("-tfp","--tfrecs_path_pattern",type=str,help="GCS bucket path pattern for tfrecords")

args = parser.parse_args()
flops = args.flops
tpu_address = args.tpu_address
tfrecs_filepath = tf.io.gfile.glob(args.tfrecs_path_pattern)

if "mf" not in flops:
    flops += "mf"

if flops not in ALLOWED_FLOPS:
    raise ValueError("Flops must be one of %s. Received: %s" % (ALLOWED_FLOPS, 
        flops.rstrip('mf')))

cfg = get_train_config()

def top1error(y_true, y_pred):
    acc = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    return 1. - acc


def make_model(flops, cfg):
    optim = tutil.get_optimizer(cfg)
    model = RegNetY(flops)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optim,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
            top1error
        ]
    )
    
    return model

cluster_resolver, strategy = tutil.connect_to_tpu(tpu_address)

if strategy:
    with strategy.scope():
        model = make_model(flops, cfg)

else:
    model = make_model(flops, cfg)


train_ds, val_ds = ImageNet(
    tfrecs_filepath,
    batch_size=128
).make_dataset()

callbacks = tutil.get_callbacks(cfg)

history = model.fit(
    train_ds,
   	epochs=cfg.total_epochs,
   	validation_data=val_ds,
   	callbacks=callbacks
)


now = datetime.now()
date_time = now.strftime("%m/%d/%Y_%H:%M")

with open(os.path.join(cfg.log_dir, 'history_%s.json' % date_time), 'a+') as f:
    json.dump(history.history, f)
