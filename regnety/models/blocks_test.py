import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import unittest
from regnety.models.blocks import SE, YBlock

tf.keras.backend.clear_session()


class blocks_test(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        self.devices = tf.config.list_physical_devices()
        self.image = tf.random.uniform((10, 20, 20, 4))
        targets = [0.0] * 3 + [1.0] + [0.0] * 6
        self.y = tf.constant([targets] * 10)

    def test_SE(self):
        block = SE(4)
        output = block(self.image)
        assert output.shape == (10, 20, 20, 4)

    def test_YBlock_stride1(self):
        if len(self.devices) > 1:
            # Grouped convs are available only on GPU
            block = YBlock(4, 4, 4, stride=1)
            output = block(self.image)
            assert output.shape == (10, 20, 20, 4)
        else:
            raise unittest.case.SkipTest("GPU not available on this machine, skipping.")

    def test_YBlock_stride2_case1(self):
        if len(self.devices) > 1:
            # Grouped convs are available only on GPU
            block = YBlock(4, 4, 8, stride=1)
            output = block(self.image)
            assert output.shape == (10, 10, 10, 8)
        else:
            raise unittest.case.SkipTest("GPU not available on this machine, skipping.")

    def test_YBlock_stride2_case2(self):
        if len(self.devices) > 1:
            # Grouped convs are available only on GPU
            block = YBlock(4, 4, 4, stride=2)
            output = block(self.image)
            assert output.shape == (10, 10, 10, 8)
        else:
            raise unittest.case.SkipTest("GPU not available on this machine, skipping.")


if __name__ == "__main__":
    unittest.main()
