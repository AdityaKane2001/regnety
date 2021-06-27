import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
import time

from dataset import imagenet



tf.keras.backend.clear_session()

imgnet = imagenet.ImageNet(
    [os.path.join('/content',i) for i in os.listdir('/content') if i.startswith('trial')],
    randaugment = True

)
ds = imgnet.make_dataset()
ds = ds.prefetch(tf.data.AUTOTUNE)
model = tf.keras.Sequential(
    [
     tf.keras.layers.InputLayer(input_shape = (224,224,3)),
     tf.keras.layers.experimental.preprocessing.Normalization(mean = 0, variance = 1),
     tf.keras.layers.Conv2D(10, 5),
     tf.keras.layers.GlobalAveragePooling2D(),
     tf.keras.layers.Dense(10, activation = 'sigmoid')
    ]
)
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)
logs = "logs/" + str(time.time())

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '2,18')

model.fit(ds,
          epochs=2,
          steps_per_epoch = 20,
          #validation_data=ds_test,
          callbacks = [tboard_callback])