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
    `stride-two 3×3 conv with w0 = 32 output filters`
    
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

class SE(tf.keras.layers.Layer):
    """
    Squeeze and Excite block. Takes se_ratio and in_filters as arguments. 

    Args:
        in_filters: Input filters. Output filters are equal to input filters 
        se_ratio: Ratio for bottleneck filters
    """

    def __init__(self, 
        in_filters: int,
        se_ratio: float = 0.25):

        self.se_filters = int(in_filters * se_ratio)
        self.out_filters = in_filters
        self.ga_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.squeeze_conv = tf.keras.layers.Conv2D(se_filters, (1,1), activation = 'relu')
        self.excite_conv = tf.keras.layers.Conv2D(out_filters, (1,1), activation = 'sigmoid')
        

    def call(self, inputs):
        # input shape: (h,w,out_filters)
        x = self.ga_pool(x) #x: (1,1,out_filters)
        x = self.squeeze_conv(x) #x: (1,1,se_filters)
        x = self.excite_conv(x) #x: (1,1,out_filters)
        x = tf.math.multiply(x, inputs) #x: (h,w,out_filters)
        return x
        

class YBlock(tf.keras.layers.Layer):
    """
    Y Block in RegNetY structure. 

    Args:
        group_width: Group width for 3x3 conv
        in_filters: Input filters in this block
        out_filters: Output filters for this block
        stride: Stride for block
    """

    def __init__(
        group_width:int,
        in_filters:int,
        out_filters:int,
        stride:int = 1
    ):
        self.group_width = group_width
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.stride = stride

        self.groups = self.out_filters // self.group_width

        self.conv1x1_1 = tf.keras.layers.Conv2D(out_filters, (1,1))
        self.se = SE(out_filters)
        self.conv1x1_2 = tf.keras.layers.Conv2D(out_filters, (1,1))

        self.bn1x1_1 = tf.keras.layers.BatchNormalization()
        self.bn3x3 = tf.keras.layers.BatchNormalization()
        self.bn1x1_2 = tf.keras.layers.BatchNormalization()

        self.relu1x1_1 = tf.keras.layers.ReLU()
        self.relu3x3 = tf.keras.layers.ReLU()
        self.relu1x1_2 = tf.keras.layers.ReLU()

        self.skip_conv = None
        self.conv3x3 = None
        self.bn_skip = None
        self.relu_skip = None

        if (in_filters != out_filters) or (stride != 1):
            self.skip_conv = tf.keras.layers.Conv2D(
                out_filters, (1, 1), stride=stride)
            self.bn_skip = tf.keras.layers.BatchNormalization()
            self.relu_skip =  = tf.keras.layers.ReLU()    
 
            self.conv3x3 = tf.keras.layers.Conv2D(
                out_filters, (3, 3), stride = stride, groups = self.groups)
 
        else:
            self.conv3x3 = tf.keras.layers.Conv2D(
                out_filters, (3, 3), stride = 1, groups = self.groups)
        
    
    def call(self, inputs):
        x = self.conv1x1_1(inputs)
        x = self.bn1x1_1(x)
        x = self.relu1x1_1(x)

        x = self.conv3x3(x)
        x = self.bn3x3(x)
        x = self.relu3x3(x)
        
        x = self.se(x)
        
        x = self.conv1x1_2(x)
        x = self.bn1x1_2(x)
        x = self.relu1x1_2(x)

        if self.skip_conv is not None:
            skip_tensor = self.skip_conv(inputs)
            skip_tensor = self.bn_skip(skip_tensor)
            skip_tensor = self.relu_skip(skip_tensor)
        
        else:
            skip_tensor = inputs
        
        x = x + skip_tensor

        return x




