import tensorflow as tf

from regnety.regnety.config import get_model_config, ALLOWED_FLOPS
from regnety.regnety.models.blocks import PreStem, Stem, Stage, Head

from typing import List, Tuple, Union


def _get_model_with_config(
    config, userdef_input_shape: Union[List, Tuple] = (None, None, 3)
):
    """
    Makes a tf.keras.Sequential model using the given config.

    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=userdef_input_shape))
    model.add(PreStem())
    model.add(Stem())

    in_channels = 32  # Output channels from Stem

    for i in range(4):  # 4 stages
        depth = config.depths[i]
        out_channels = config.widths[i]
        group_width = config.group_width

        model.add(Stage(depth, group_width, in_channels, out_channels, stage_num=i))

        in_channels = out_channels

    model.add(Head(config.num_classes))

    return model


def RegNetY(
    flops: str = "", input_shape: Union[List, Tuple] = None
) -> tf.keras.Sequential:
    """
    Instantiates a RegNetY instance based on flops and input_shape furnished by user.

    Args:
        flops: Flops of the model eg. "400MF" (Processing one image requires
            400 million floating point operations (multiplication, addition))
        input_shape: A python list or tuple denoting the shape input. Please omit the
            batch dimension. eg. (256, 256, 3). Must be greater than or equal to 224.

    Returns:
        A tf.keras.Sequential with RegNetY architecture.
    """

    if flops not in ALLOWED_FLOPS:
        raise ValueError("`flops` must be one of " + str(ALLOWED_FLOPS))

    if input_shape is None:
        userdef_input_shape = (None, None, 3)
    else:
        userdef_input_shape = input_shape

    if userdef_input_shape[0] is not None:
        if any([i < 224 for i in userdef_input_shape[:-1]]):
            raise ValueError(
                "All non-channel dimensions in `input_shape`"
                " must be greater than or equal to 224."
            )

    try:
        assert len(userdef_input_shape) == 3
    except:
        raise ValueError(
            "Input shape is invalid. Please enter input shape " " as (height, width, 3)"
        )

    config = get_model_config(flops)
    model = _get_model_with_config(config)
    return model
