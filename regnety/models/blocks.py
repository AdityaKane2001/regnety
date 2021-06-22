"""Has building blocks of RegNetY models"""

import tensorflow as tf


# TODO:
# 1. Stem
# 2. Body
#     2.1 Block
#         2.1.1 SE
# 3. Head
# 4. RegNetY
# Reference: https://tinyurl.com/dcc5c5p8

class Stem(tf.keras.layers.Layer):
    """Class to initiate stem architecture from the paper: 
    `stride-two 3×3 conv with w0 = 32 output channels`
    
    Args:
        None, stem is common to all models
    """

    def __init__(self):

        self.conv3x3 =  tf.keras.layers.Conv2D(32, (3,3), strides = 2)
        self.bn = tf.keras.layers.BatchNormalization(
            momentum = 0.9, epsilon = 0.00001
        )
        self.act = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv3x3(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x

