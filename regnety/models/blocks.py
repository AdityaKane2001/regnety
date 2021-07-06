"""Contains the building blocks of RegNetY models."""

import tensorflow as tf

from tensorflow.keras import layers
from typing import Union

# Contains:
# 0. PreStem 
# 1. Stem
# 2. Body
#     2.1 Block
#         2.1.1 SE
# 3. Head
# 4. RegNetY
# Reference: https://arxiv.org/pdf/2003.13678.pdf

_MEAN = tf.constant([0.485, 0.456, 0.406])
_VAR = tf.constant([0.052441, 0.050176, 0.050625])

class PreStem(layers.Layer):
    """Contains preprocessing layers which are to be included in the model.
    
    Args: 
        mean: Mean to normalize to
        variance: Variance to normalize to
        crop_size: Size to take random crop to before resizing to 224x224 
    """

    def __init__(self,
        mean: tf.Tensor = _MEAN,
        variance: tf.Tensor = _VAR,
        crop_size: int = 320
    ):
        super(PreStem, self).__init__()
        self.crop_size = crop_size
        self.mean = mean
        self.var = variance

        self.rand_crop = layers.experimental.preprocessing.RandomCrop(
            self.crop_size, self.crop_size, name = 'prestem_random_crop'
        )
        self.resize = layers.experimental.preprocessing.Resizing(
            224, 224, name = 'prestem_resize'
        )
        self.norm = layers.experimental.preprocessing.Normalization(
            mean = self.mean, variance = self.var, name = 'prestem_normalize'
        )
    
    def call(self, inputs):
        x = self.rand_crop(inputs)
        x = self.resize(x)
        x = self.norm(x)
        return x
    
    def get_config(self):
        """
        
        """


class Stem(layers.Layer):
    """Class to initiate stem architecture from the paper (see `Reference` 
    above): `stride-two 3×3 conv with w0 = 32 output filters`.
    
    Args:
        None, stem is common to all models
    """

    def __init__(self):
        super(Stem, self).__init__()
        self.conv3x3 =  layers.Conv2D(32, (3,3), strides = 2)
        self.bn = layers.BatchNormalization(
            momentum = 0.9, epsilon = 0.00001
        )
        self.act = layers.ReLU()

    def call(self, inputs):
        x = self.conv3x3(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x

class SE(layers.Layer):
    """
    Squeeze and Excite block. Takes se_ratio and in_filters as arguments. 

    Args:
        in_filters: Input filters. Output filters are equal to input filters 
        se_ratio: Ratio for bottleneck filters
        name_prefix: prefix to be given to name
    """

    def __init__(self, 
        in_filters: int,
        se_ratio: float = 0.25,
        name_prefix: str = ''
    ):
        super(SE, self).__init__(name = name_prefix + 'SE')
        
        self.se_filters = int(in_filters * se_ratio)
        self.out_filters = in_filters
        self.pref = name_prefix

        self.ga_pool = layers.GlobalAveragePooling2D(
            name = self.pref + '_global_avg_pool'
        )
        self.squeeze_dense = layers.Dense(
            self.se_filters, activation = 'relu', 
            name = self.pref + '_squeeze_dense'
        )
        self.excite_dense = layers.Dense(
            self.out_filters,  activation = 'sigmoid', 
            name = self.pref + '_excite_dense'
        )
        
        
    def call(self, inputs):
        # input shape: (h,w,out_filters)
        x = self.ga_pool(inputs) # x: (out_filters)
        x = self.squeeze_dense(x) # x: (se_filters)
        x = self.excite_dense(x) # x: (out_filters)
        x = tf.reshape(x, [-1,1,1,self.out_filters])
        x = tf.math.multiply(x, inputs) # x: (h,w,out_filters)
        return x
        

class YBlock(layers.Layer):
    """
    Y Block in RegNetY structure. 
    IMPORTANT: Grouped convolutions are only supported by keras on GPU. 

    Args:
        group_width: Group width for 3x3 conv, in_filters and out_filters must
             be divisible by this.
        in_filters: Input filters in this block
        out_filters: Output filters for this block
        stride: Stride for block
        name_prefix: prefix for name
    """

    def __init__(self,
        group_width:int,
        in_filters:int,
        out_filters:int,
        stride:int = 1,
        name_prefix: str = ''
    ):
        super(YBlock, self).__init__(name = name_prefix)

        self.group_width = group_width
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.stride = stride
        self.pref = name_prefix

        self.groups = self.out_filters // self.group_width

        self.conv1x1_1 = layers.Conv2D(out_filters, (1,1), 
            name = self.pref + '_conv1x1_1'
        )
        self.se = SE(out_filters, name_prefix = self.pref)
        self.conv1x1_2 = layers.Conv2D(out_filters, (1,1), 
            name = self.pref + '_conv1x1_2'
        )

        self.bn1x1_1 = layers.BatchNormalization(
            momentum = 0.9, epsilon = 0.00001, 
            name = self.pref + '_bn1x1_1'
        )
        self.bn3x3 = layers.BatchNormalization(
            momentum = 0.9, epsilon = 0.00001, 
            name = self.pref + '_bn3x3'
        )
        self.bn1x1_2 = layers.BatchNormalization(
            momentum = 0.9, epsilon = 0.00001, 
            name = self.pref + '_bn1x1_2'
        )

        self.relu1x1_1 = layers.ReLU(name = self.pref + '_relu1x1_1')
        self.relu3x3 = layers.ReLU(name = self.pref + '_relu3x3')
        self.relu1x1_2 = layers.ReLU(name = self.pref + '_relu1x1_2')

        self.skip_conv = None
        self.conv3x3 = None
        self.bn_skip = None
        self.relu_skip = None

        if (in_filters != out_filters) or (stride != 1):
            self.skip_conv = layers.Conv2D(
                out_filters, (1, 1), strides=2, name = self.pref + '_conv_skip')
            self.bn_skip = layers.BatchNormalization(
                name = self.pref + '_bn_skip'
            )
            self.relu_skip = layers.ReLU(name = self.pref + '_relu_skip')    
 
            self.conv3x3 = layers.Conv2D(
                out_filters, (3, 3), strides = 2, groups = self.groups, 
                padding = 'same', name = self.pref + '_conv3x3')
 
        else:
            self.conv3x3 = layers.Conv2D(
                out_filters, (3, 3), strides = 1, groups = self.groups, 
                padding = 'same', name = self.pref + '_conv3x3' )
        
    
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
    


class Stage(layers.Layer):
    """
    Class for RegNetY stage. A single stage consists of `depth` number of 
    YBlocks. Such four stages are connected sequantially to create `body` 
    of the model. For more information, refer to the paper (see `Reference` at 
    the top of this module).

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
        out_filters:int,
        stage_num: int = 0
    ):
        super(Stage, self).__init__(name = 'Stage_' + str(stage_num))
        
        self.depth = depth
        self.pref = 'Stage_' + str(stage_num) + '_'

        self.stage = []

        self.stage.append(YBlock(
            group_width, in_filters, out_filters, stride = 2,
            name_prefix = self.pref + 'YBlock_0'
        ))

        for block_num in range(depth - 1):
            self.stage.append(
                YBlock(group_width, out_filters, out_filters, stride = 1, 
                name_prefix = self.pref + 'YBlock_' + str(block_num + 1))
            )

    def call(self, inputs):
        x = inputs
        for i in range(self.depth):
            x = self.stage[i](x)
        
        return x


class Head(layers.Layer):
    """
    Head for all RegNetY models.

    Args:
        num_classes: Integer specifying number of classes of data. 
    """
    def __init__(self, num_classes):
        super(Head, self).__init__(name = 'Head')

        self.gap = layers.GlobalAveragePooling2D(name = 'Head_global_avg_pool')
        self.fc = layers.Dense(num_classes, name = 'Head_fc')
    
    def call(self, inputs):
        x = self.gap(inputs)
        x = self.fc(x)
        return x 
