import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import unittest
from regnety.regnety.models.blocks import SE, YBlock

tf.keras.backend.clear_session()

class blocks_test(unittest.TestCase):
    
    def setUp(self):
        tf.keras.backend.clear_session()
        self.devices = tf.config.list_physical_devices()
        self.image = tf.random.uniform((10,20,20,4))
        targets = [0.0]*3 + [1.] + [0.0]*6
        self.y = tf.constant([targets]*10)

    def test_SE(self):
        model = tf.keras.Sequential([
            SE(4),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, activation= 'sigmoid')
            ])
        model.compile(optimizer = 'adam',
            metrics=['accuracy'], loss = 'categorical_crossentropy')
        model.fit(self.image,self.y,verbose=0)

    def test_YBlock_stride1(self):
        if len(self.devices) > 1:
            # Grouped convs are available only on GPU
            yblock = YBlock(4, 4, 8, stride = 1)
            model = tf.keras.Sequential([
                yblock,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(10, activation= 'sigmoid')
            ])
            model.compile(optimizer = 'adam',
                metrics=['accuracy'], loss = 'categorical_crossentropy')
            model.fit(self.image,self.y,verbose=0)
        else:
            raise unittest.case.SkipTest("GPU not available on this machine, skipping.")

    def test_YBlock_stride2(self):
        if len(self.devices) > 1:
            # Grouped convs are available only on GPU
            yblock = YBlock(4, 4, 8, stride = 2)
            model = tf.keras.Sequential([
                yblock,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(10, activation= 'sigmoid')
            ])
            model.compile(optimizer = 'adam',
                metrics=['accuracy'], loss = 'categorical_crossentropy')
            model.fit(self.image,self.y,verbose=0)
        else:
            raise unittest.case.SkipTest("GPU not available on this machine, skipping.")




if __name__ == '__main__':
    unittest.main()

