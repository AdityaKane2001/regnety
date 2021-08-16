"""Contains the building blocks of RegNetY models."""


import tensorflow as tf

from tensorflow.keras import layers
from regnety.models.utils import ConvInitializer, DenseInitializer

# Contains:
# 0. PreStem
# 1. Stem
# 2. Body
#     2.1 Block
#         2.1.1 SE
# 3. Head
# 4. RegNetY
# Reference: https://arxiv.org/pdf/2003.13678.pdf

# ImageNet mean and variance
_MEAN = tf.constant([0.485, 0.456, 0.406])
_VAR = tf.constant([0.052441, 0.050176, 0.050625])



class PreStem(layers.Layer):
    """Contains preprocessing layers which are to be included in the model.
    
    Args: 
        mean: Mean to normalize to
        variance: Variance to normalize to
    """

    def __init__(self,
                 mean: tf.Tensor = _MEAN,
                 variance: tf.Tensor = _VAR,
                 ):
        super(PreStem, self).__init__(name='PreStem')
        self.mean = mean
        self.var = variance

        self.rescale = layers.experimental.preprocessing.Rescaling(
            scale=1./255., name="prestem_rescale"
        )
        self.resize = layers.experimental.preprocessing.Resizing(
            224, 224, name='prestem_resize'
        )
        self.norm = layers.experimental.preprocessing.Normalization(
            mean=self.mean, variance=self.var, name='prestem_normalize'
        )

    def call(self, inputs):
        x = self.resize(inputs)
        x = self.rescale(x)
        x = self.norm(x)
        return x

    def get_config(self):

        config = super(PreStem, self).get_config()
        config.update({
            'mean': [0.485, 0.456, 0.406],
            'variance': [0.052441, 0.050176, 0.050625]
        })
        return config


class Stem(layers.Layer):
    """Class to initiate stem architecture from the paper (see `Reference` 
    above): `stride-two 3×3 conv with w0 = 32 output filters`.
    
    Args:
        None, stem is common to all models
    """

    def __init__(self):
        super(Stem, self).__init__(name='Stem')
        self.conv3x3 = layers.Conv2D(
            32, (3, 3), strides=2, use_bias=False, 
            kernel_initializer=ConvInitializer()
        )
        self.bn = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5
        )
        self.act = layers.ReLU()

    def call(self, inputs):
        x = self.conv3x3(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x

    def get_config(self):

        config = super(Stem, self).get_config()
        return config


class SE(layers.Layer):
    """
    Squeeze and Excite block. Takes se_ratio and in_filters as arguments. 
    Arxiv link: https://arxiv.org/abs/1709.01507?spm=a2c41.13233144.0.0

    Args:
        in_filters: Input filters. Output filters are equal to input filters 
        se_ratio: Ratio for bottleneck filters
        name_prefix: prefix to be given to name
    """

    def __init__(
        self, in_filters: int = 0, se_ratio: float = 0.25, name_prefix: str = ""
    ):
        super(SE, self).__init__(name=name_prefix + "SE")

        self.in_filters = in_filters
        self.se_ratio = se_ratio
        self.se_filters = int(self.in_filters * self.se_ratio)
        self.out_filters = in_filters
        self.pref = name_prefix

        self.ga_pool = layers.GlobalAveragePooling2D(
            name=self.pref + "_global_avg_pool"
        )
        self.squeeze_reshape = layers.Reshape((1, 1, self.out_filters))
        self.squeeze_conv = layers.Conv2D(
            self.se_filters, (1, 1), activation="relu",
            name=self.pref + "_squeeze_conv",
            kernel_initializer=ConvInitializer()
        )
        self.excite_conv = layers.Conv2D(
            self.out_filters, (1, 1), activation="sigmoid", 
            name=self.pref + "_excite_conv",
            kernel_initializer=ConvInitializer()
        )

    def call(self, inputs):
        # input shape: (h,w,out_filters)
        x = self.ga_pool(inputs)  # x: (out_filters)
        x = self.squeeze_reshape(x)  # x: (1, 1, out_filters)
        x = self.squeeze_conv(x)  # x: (1, 1, se_filters)
        x = self.excite_conv(x)  # x: (1, 1, out_filters)
        x = tf.math.multiply(x, inputs)  # x: (h,w,out_filters)
        return x

    def get_config(self):

        config = super(SE, self).get_config()
        config.update({
            'in_filters': self.in_filters,
            'se_ratio': self.se_ratio,
            'name_prefix': self.pref
        })
        return config


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
                 group_width: int = 0,
                 in_filters: int = 0,
                 out_filters: int = 0,
                 stride: int = 1,
                 name_prefix: str = ''
                 ):
        super(YBlock, self).__init__(name=name_prefix)

        self.group_width = group_width
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.stride = stride
        self.pref = name_prefix

        self.groups = self.out_filters // self.group_width

        self.conv1x1_1 = layers.Conv2D(out_filters, (1, 1),
                                       name=self.pref + '_conv1x1_1',
                                       use_bias=False,
                                       kernel_initializer=ConvInitializer()
                                       )
        self.se = SE(out_filters, name_prefix=self.pref + '_')
        self.conv1x1_2 = layers.Conv2D(out_filters, (1, 1),
                                       name=self.pref + '_conv1x1_2',
                                       use_bias=False,
                                       kernel_initializer=ConvInitializer()
                                       )

        self.bn1x1_1 = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5,
            name=self.pref + '_bn1x1_1'
        )
        self.bn3x3 = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5,
            name=self.pref + '_bn3x3'
        )
        self.bn1x1_2 = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5,
            name=self.pref + '_bn1x1_2'
        )

        self.relu1x1_1 = layers.ReLU(name=self.pref + '_relu1x1_1')
        self.relu3x3 = layers.ReLU(name=self.pref + '_relu3x3')
        self.relu = layers.ReLU(name=self.pref + '_relu')

        self.skip_conv = None
        self.conv3x3 = None
        self.bn_skip = None

        if (in_filters != out_filters) or (stride != 1):
            self.skip_conv = layers.Conv2D(
                out_filters, (1, 1), strides=2, name=self.pref + '_conv_skip',
                kernel_initializer=ConvInitializer(),
                use_bias=False)
            self.bn_skip = layers.BatchNormalization(
                name=self.pref + '_bn_skip',  momentum=0.9, epsilon=1e-5,
            )

            self.conv3x3 = layers.Conv2D(
                out_filters, (3, 3), strides=2, groups=self.groups,
                kernel_initializer=ConvInitializer(),
                padding='same', name=self.pref + '_conv3x3', use_bias=False)

        else:
            self.conv3x3 = layers.Conv2D(
                out_filters, (3, 3), strides=1, groups=self.groups,
                kernel_initializer=ConvInitializer(),
                padding='same', name=self.pref + '_conv3x3', use_bias=False)

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

        if self.skip_conv is not None:
            skip_tensor = self.skip_conv(inputs)
            skip_tensor = self.bn_skip(skip_tensor)

        else:
            skip_tensor = inputs

        x = self.relu(x + skip_tensor)

        return x

    def get_config(self):

        config = super(YBlock, self).get_config()
        config.update({
            'group_width': self.group_width,
            'in_filters': self.in_filters,
            'out_filters': self.out_filters,
            'stride': self.stride,
            'name_prefix': self.pref
        })
        return config


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
                 depth: int = 0,
                 group_width: int = 0,
                 in_filters: int = 0,
                 out_filters: int = 0,
                 stage_num: int = 0
                 ):
        super(Stage, self).__init__(name='Stage_' + str(stage_num))

        self.depth = depth
        self.group_width = group_width
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.stage_num = stage_num

        self.pref = 'Stage_' + str(stage_num) + '_'

        self.stage = []

        self.stage.append(YBlock(
            group_width, in_filters, out_filters, stride=2,
            name_prefix=self.pref + 'YBlock_0'
        ))

        for block_num in range(depth - 1):
            self.stage.append(
                YBlock(group_width, out_filters, out_filters, stride=1,
                       name_prefix=self.pref + 'YBlock_' + str(block_num + 1))
            )

    def call(self, inputs):
        x = inputs
        for i in range(self.depth):
            x = self.stage[i](x)

        return x

    def get_config(self):
        config = super(Stage, self).get_config()
        config.update({
            'depth': self.depth,
            'group_width': self.group_width,
            'in_filters': self.in_filters,
            'out_filters': self.out_filters,
            'stage_num': self.stage_num
        })
        return config


class Head(layers.Layer):
    """
    Head for all RegNetY models. Returns logits.

    Args:
        num_classes: Integer specifying number of classes of data. 
    """

    def __init__(self, num_classes):
        super(Head, self).__init__(name='Head')

        self.num_classes = num_classes
        self.gap = layers.GlobalAveragePooling2D(name='Head_global_avg_pool')
        self.fc = layers.Dense(
            self.num_classes, 
            name='Head_fc', 
            kernel_initializer=DenseInitializer())

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.fc(x)
        return x

    def get_config(self):
        config = super(Head, self).get_config()
        config.update({
            'num_classes': self.num_classes
        })
        return config
