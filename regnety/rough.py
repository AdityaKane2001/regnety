import tensorflow as tf
import tensorflow_addons as tfa
import math
import wandb
from wandb.keras import WandbCallback
import regnety.regnety.utils.model_utils as mutil
from regnety.regnety.models import RegNetY
from regnety.regnety.dataset.imagenet import ImageNet
from regnety.regnety.utils import train_utils as tutil
from regnety.regnety.config.config import (
    get_train_config,
    get_custom_train_config,
    get_preprocessing_config,
    ALLOWED_FLOPS
)


PI = math.pi

def preprocess(image, target):
    aug_images = tf.image.resize(image, (224, 224))
    aug_images = tf.cast(aug_images, tf.float32)
    aug_images = aug_images / 127.5
    aug_images = aug_images - 1
    return aug_images, target

tf.keras.backend.clear_session()

tfrecs_filepath = tf.io.gfile.glob("gs://adityakane-imagenette-tfrecs/*.tfrecord")

prep_cfg = get_preprocessing_config( 
    tfrecs_filepath=tfrecs_filepath,
    batch_size=1024,
    image_size=512,
    crop_size=224,
    resize_to_size=320,
    augment_fn=preprocess,
    num_classes=1000,
    percent_valid=11,
    cache_dir="gs://adityakane-train/cache/",
    color_jitter=False,
    scale_to_unit=False
) 


train_ds, val_ds = ImageNet(prep_cfg).make_dataset()

def half_cos_schedule(epoch, lr):
    # Taken from pycls/pycls/core/optimizer.py, since not clear from paper.
    if epoch < 5:
        new_lr = 0.5 * (1.0 + tf.math.cos(PI * epoch /
            100)) * 0.8
        alpha = epoch / 5
        warmup_factor = 0.1 * (1. - alpha) + alpha
        return new_lr * warmup_factor
    else:
        new_lr = new_lr = 0.5 * (1.0 + tf.math.cos(PI * epoch /
            100)) * 0.8
        return new_lr


cluster_resolver, strategy = tutil.connect_to_tpu(None)

with strategy.scope():
    model = tf.keras.applications.ResNet50V2(
        include_top=True,
        weights=None,
    )
    optim = tfa.optimizers.SGDW(
            weight_decay=5e-4,
            learning_rate=0.8,
            momentum=0.9,
            nesterov=True
        )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=optim,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ]
    )


wandb.init(entity='compyle', project='regnety',
           job_type='train', name= 'ResNet50V2')


trial_callbacks = [
    tf.keras.callbacks.LearningRateScheduler(half_cos_schedule),
    WandbCallback()
]


history = model.fit(
    train_ds,
   	epochs=100,
   	validation_data=val_ds,
   	callbacks=trial_callbacks
)