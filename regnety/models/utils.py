"""Contains utilities for instantiating model architecture."""

import tensorflow as tf

class ConvInitializer(tf.keras.initializers.Initializer):
    """
    Initializer class for convolution layers.
    
    Args: None
    """
    def __init__(self):
        super(ConvInitializer, self).__init__()

    def __call__(self, shape, dtype, **kwargs):
        fan_out = tf.cast(tf.math.reduce_prod(shape) / shape[2], tf.float32)
        return tf.random.normal(
            shape,
            mean=tf.cast(0.0, tf.float32),
            stddev=tf.cast(tf.math.sqrt(2.0 / fan_out), tf.float32),
            dtype=dtype
        )

class DenseInitializer(tf.keras.initializers.Initializer):
    """
    Initializer class for dense layers.
    
    Args: None
    """

    def __init__(self):
        super(DenseInitializer, self).__init__()

    def __call__(self, shape, dtype, **kwargs):
        return tf.random.normal(
            shape,
            mean=0.0,
            stddev=0.01,
            dtype=dtype
        )
