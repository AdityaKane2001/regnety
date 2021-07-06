import tensorflow as tf
from regnety.regnety.models.model import RegNetY

def plot_model(
    regnety_instance: regnety.regnety.models.model.RegNetY,
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

def get_layer_names_dict(regnety_instance: regnety.regnety.models.model.RegNetY):
    """
    Returns a detailed dict containing names of all stage, blocks, layers

    Args: 
        regnety_instance: A RegNetY instance
    
    Returns:
        Dict which contains names of all layers in a hierarchical manner 
    """

    pass


