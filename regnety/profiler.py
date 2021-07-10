import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from PIL import Image
#%matplotlib inline
import matplotlib.pyplot as plt


from dataset import imagenet



tf.keras.backend.clear_session()

imgnet = imagenet.ImageNet(
    tf.io.gfile.glob('/content/*.tfrecord'),

    batch_size = 128
)

ds = imgnet.make_dataset()

ds = ds.prefetch(tf.data.AUTOTUNE)
k=0
for i in ds:
    im = i[0][0].numpy() / 255.
    k+=1
    img = plt.imshow(im)
    img.set_cmap('hot')
    plt.axis('off')
    plt.savefig(os.path.join('/content','augmented_'+str(k)+'.jpeg'), bbox_inches='tight')
    break

    


# model = tf.keras.Sequential(
#     [
#      tf.keras.layers.InputLayer(input_shape = (512,512,3)),
#      tf.keras.layers.experimental.preprocessing.Normalization(mean = 0, variance = 1),
#      tf.keras.layers.Conv2D(10, 5),
#      tf.keras.layers.GlobalAveragePooling2D(),
#      tf.keras.layers.Dense(10, activation = 'sigmoid')
#     ]
# )
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=tf.keras.optimizers.Adam(0.001),
#     metrics=['accuracy']
# )
# logs = "logs/" + '128_no_cache'

# tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
#                                                  histogram_freq = 1,
#                                                  profile_batch = '2,29')

# model.fit(ds,
#           epochs=3,
#           steps_per_epoch = 10,
#           #validation_data=ds_test,
#           callbacks = [tboard_callback])