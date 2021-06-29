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
        super(Stem, self).__init__()
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
        se_ratio: float = 0.25
    ):
        super(SE, self).__init__()
        
        self.se_filters = int(in_filters * se_ratio)
        self.out_filters = in_filters
        self.ga_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.squeeze_dense = tf.keras.layers.Dense(self.se_filters, activation = 'relu')
        self.excite_dense = tf.keras.layers.Dense(self.out_filters,  activation = 'sigmoid')
        
    def call(self, inputs):
        # input shape: (h,w,out_filters)
        x = self.ga_pool(inputs) #x: (out_filters)
        x = self.squeeze_dense(x) #x: (se_filters)
        x = self.excite_dense(x) #x: (out_filters)
        x = tf.reshape(x, [-1,1,1,self.out_filters])
        x = tf.math.multiply(x, inputs) #x: (h,w,out_filters)
        return x
        

class YBlock(tf.keras.layers.Layer):
    """
    Y Block in RegNetY structure. 
    IMPORTANT: Grouped convolutions are only supported by keras on GPU. 

    Args:
        group_width: Group width for 3x3 conv, in_filters and out_filters must
             be divisible by this.
        in_filters: Input filters in this block
        out_filters: Output filters for this block
        stride: Stride for block
    """

    def __init__(self,
        group_width:int,
        in_filters:int,
        out_filters:int,
        stride:int = 1
    ):
        super(YBlock, self).__init__()

        self.group_width = group_width
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.stride = stride

        self.groups = self.out_filters // self.group_width

        self.conv1x1_1 = tf.keras.layers.Conv2D(out_filters, (1,1))
        self.se = SE(out_filters)
        self.conv1x1_2 = tf.keras.layers.Conv2D(out_filters, (1,1))

        self.bn1x1_1 = tf.keras.layers.BatchNormalization(
            momentum = 0.9, epsilon = 0.00001)
        self.bn3x3 = tf.keras.layers.BatchNormalization(
            momentum = 0.9, epsilon = 0.00001)
        self.bn1x1_2 = tf.keras.layers.BatchNormalization(
            momentum = 0.9, epsilon = 0.00001)

        self.relu1x1_1 = tf.keras.layers.ReLU()
        self.relu3x3 = tf.keras.layers.ReLU()
        self.relu1x1_2 = tf.keras.layers.ReLU()

        self.skip_conv = None
        self.conv3x3 = None
        self.bn_skip = None
        self.relu_skip = None

        if (in_filters != out_filters) or (stride != 1):
            self.skip_conv = tf.keras.layers.Conv2D(
                out_filters, (1, 1), strides=2)
            self.bn_skip = tf.keras.layers.BatchNormalization()
            self.relu_skip = tf.keras.layers.ReLU()    
 
            self.conv3x3 = tf.keras.layers.Conv2D(
                out_filters, (3, 3), strides = 2, groups = self.groups, 
                padding = 'same')
 
        else:
            self.conv3x3 = tf.keras.layers.Conv2D(
                out_filters, (3, 3), strides = 1, groups = self.groups, 
                padding = 'same')
        
    
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
    


class Stage(tf.keras.layers.Layer):
    """
    Class for RegNetY stage. 

    Args:
        depth: Depth of stage, number of blocks to use
        group_width: Group width of all blocks in  this stage
        in_filters: Input filters to this stage
        out_filters: Output filters from this stage
        
    """

    def __init__(self,
        depth:int,
        group_width:int,
        in_filters:int,
        out_filters:int
    ):
        super(Stage, self).__init__()
        
        self.depth = depth

        self.stage = []

        self.stage.append(YBlock(group_width, in_filters, out_filters, stride = 2))

        for _ in range(depth - 1):
            self.stage.append(
                YBlock(group_width, out_filters, out_filters, stride = 1)
            )

    def call(self, inputs):
        x = inputs
        for i in range(self.depth):
            x = self.stage[i](x)
        
        return x


class Head(tf.keras.layers.Layer):
    """
    Head for all RegNetY models.

    Args:
        num_classes: Integer specifying number of classes of data. 
    """
    def __init__(self, num_classes):
        super(Head, self).__init__()

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes, activation = 'sigmoid')
    
    def call(self, inputs):
        x = self.gap(inputs)
        x = self.fc(x)
        return x 



