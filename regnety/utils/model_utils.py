import tensorflow as tf
from regnety.regnety.models.model import RegNetY
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)

def plot_model(
    regnety_instance: RegNetY,
    imgpath: str = "model.png" ):

    """
     Plots model using keras utility

     Args:
        regnety_instance: A RegNetY instance
    
    Returns: None
    """
    tf.keras.utils.plot_model(
                regnety_instance.get_model(),
                to_file=imgpath,
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
            )

def get_layer_names_dict(regnety_instance:RegNetY):
    """
    Returns a detailed dict containing names of all stage, blocks, layers

    Args: 
        regnety_instance: A RegNetY instance
    
    Returns:
        Dict which contains names of all layers in a hierarchical manner 
    """

    pass


def get_flops(regnety_instance: RegNetY):
    """
    Calculate FLOPS for tf.keras.Model or tf.keras.Sequential .
    Ignore operations used in only training mode such as Initialization.
    Use tf.profiler of tensorflow v1 api.

    Args:
        regnety_instance: A regnety.models.model.RegNetY instance

    Returns:
        Tuple containing total float ops and paramenters
    """
    model = regnety_instance.get_model()
    if not isinstance(model, (tf.keras.Sequential, tf.keras.Model)):
        raise KeyError(
            "model arguments must be tf.keras.Model or tf.keras.Sequential instanse"
        )
    inputs = [
        tf.TensorSpec(regnety_instance.userdef_input_shape,tf.float32) 
    ]
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )
    return flops.total_float_ops//2, flops.parameters