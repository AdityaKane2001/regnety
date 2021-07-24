from regnety.regnety.config.config import (
    get_train_config,
    get_preprocessing_config,
    ALLOWED_FLOPS
)
from regnety.regnety.utils import train_utils as tutil
from regnety.regnety.dataset.imagenet import ImageNet
from regnety.regnety.models.model import RegNetY
from wandb.keras import WandbCallback
from datetime import datetime
import tensorflow_addons as tfa
import keras_tuner as kt
import wandb
import regnety
import json
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


FLOPS = "200mf"

hp = kt.HyperParameters()

tfrecs_filepath = tf.io.gfile.glob(
    "gs://adityakane-imagenette-tfrecs/*.tfrecord")
tfrecs_filepath.sort()
train_tfrecs_filepath = tfrecs_filepath[:-1]
val_tfrecs_filepath = [tfrecs_filepath[-1]]


train_prep_cfg = get_preprocessing_config(
    tfrecs_filepath=train_tfrecs_filepath,
    batch_size=1024,
    image_size=512,
    crop_size=224,
    resize_pre_crop=224,
    augment_fn="default",
    num_classes=1000,
    cache_dir="gs://ak-europe-train/cache/",
    color_jitter=False,
    scale_to_unit=True
)

val_prep_cfg = get_preprocessing_config(
    tfrecs_filepath=val_tfrecs_filepath,
    batch_size=1024,
    image_size=512,
    crop_size=224,
    resize_pre_crop=224,
    augment_fn="val",
    num_classes=1000,
    cache_dir="gs://ak-europe-train/cache/",
    color_jitter=False,
    scale_to_unit=True
)

cluster_resolver, tpu_strat = tutil.connect_to_tpu()


def make_model(hp, strategy=tpu_strat):
    hp_weight_decay = hp.Choice("weight_decay",
                                [5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
    hp_base_lr = hp.Choice("base_lr",
                           [5e-4, 1e-3, 5e-3])
    optim = tfa.optimizers.AdamW(
        weight_decay=hp_weight_decay,
        learning_rate=hp_base_lr
    )

    with strategy.scope():
        model = regnety.regnety.models.model.RegNetY(FLOPS)
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=optim,
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.TopKCategoricalAccuracy(
                    5, name="top-5-accuracy"),
                tutil.top1error
            ]
        )
    return model


wandb.init(entity="compyle", project="regnety",
           job_type="train", name="Tuner200MF")

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
train_ds = ImageNet(train_prep_cfg).make_dataset()
val_ds = ImageNet(val_prep_cfg).make_dataset()
tuner = kt.BayesianOptimization(
    make_model,
    "val_accuracy",
    20,
    directory="gs://ak-europe-train/tunercache",
    project_name="200MF",
    distribution_strategy=tpu_strat)

tuner.search(train_ds, epochs=75, validation_data=val_ds,
             callbacks=[stop_early, WandbCallback()])
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%Hh%Mm")

bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]

print(bestHP)

with tf.io.gfile.GFile(os.path.join("gs://ak-europe-train/tunercache/", "bestHP_%s.json" % date_time), "a+") as f:
   json.dump(str(bestHP), f)
