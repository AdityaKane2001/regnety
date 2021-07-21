import tensorflow as tf
import tensorflow_addons as tfa
import math
import os
import regnety

PI = math.pi


def get_optimizer(cfg: regnety.regnety.config.config.TrainConfig):
    if cfg.optimizer == "sgd":
        return tfa.optimizers.SGDW(
            weight_decay=cfg.weight_decay,
            learning_rate=cfg.base_lr,
            momentum=cfg.momentum,
            nesterov=True
        )

    elif cfg.optimizer == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=cfg.base_lr,
        )

    elif cfg.optimizer == "adamw":
        return tfa.optimizers.AdamW(
            weight_decay=cfg.weight_decay,
            learning_rate=cfg.base_lr
        )

    else:
        raise NotImplementedError(
            f"Optimizer choice not supported: {cfg.optimizer}"
        )


def get_train_schedule(cfg: regnety.regnety.config.config.TrainConfig):
    if cfg.lr_schedule == "half_cos":
        def half_cos_schedule(epoch, lr):
            # Taken from pycls/pycls/core/optimizer.py, since not clear from paper.
            if epoch < cfg.warmup_epochs:
                new_lr = 0.5 * (1.0 + tf.math.cos(PI * epoch /
                    cfg.total_epochs)) * cfg.base_lr
                alpha = epoch / cfg.warmup_epochs
                warmup_factor = cfg.warmup_factor * (1. - alpha) + alpha
                return new_lr * warmup_factor
            else:
                new_lr = 0.5 * (1.0 + tf.math.cos(PI * epoch /
                    cfg.total_epochs)) * cfg.base_lr
                return new_lr

        return half_cos_schedule

    elif cfg.lr_schedule == "constant":
        return cfg.base_lr


def get_callbacks(cfg):
    lr_callback = tf.keras.callbacks.LearningRateScheduler(get_train_schedule(cfg))
    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=cfg.log_dir, histogram_freq=1)  # profile_batch="0,1023"
    best_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(cfg.model_dir, "best_model_{epoch:02d}-{val_loss:.2f}.h5"),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True)
    all_models_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            cfg.model_dir, "all_model_{epoch:02d}-{val_loss:.2f}.h5"),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=False)

    return [
        lr_callback,
        tboard_callback,
        best_model_checkpoint_callback,
        all_models_checkpoint_callback
    ]


def top1error(y_true, y_pred):
    acc = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    return 1. - acc


def make_model(flops, train_cfg):
    optim = get_optimizer(train_cfg)
    model = regnety.regnety.models.model.RegNetY(flops)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=optim,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
            top1error
        ]
    )

    return model

def connect_to_tpu(tpu_address: str = None):
    if tpu_address is not None:  # When using GCP
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_address)
        if tpu_address not in ("", "local"):
            tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        print("Running on TPU ", cluster_resolver.master())
        print("REPLICAS: ", strategy.num_replicas_in_sync)
        return cluster_resolver, strategy
    else:                           # When using Colab or Kaggle
        try:
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            strategy = tf.distribute.TPUStrategy(cluster_resolver)
            print("Running on TPU ", cluster_resolver.master())
            print("REPLICAS: ", strategy.num_replicas_in_sync)
            return cluster_resolver, strategy
        except:
            print("WARNING: No TPU detected.")
            mirrored_strategy = tf.distribute.MirroredStrategy()
            return None, mirrored_strategy
