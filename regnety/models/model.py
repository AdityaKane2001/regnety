import tensorflow as tf

from regnety.regnety.config import get_model_config, ALLOWED_FLOPS
from blocks import Stem, Stage, Head

class RegNetY(tf.keras.Model):
    """
    RegNetY model class. Subclassed from tf.keras.Model. Instantiates a randomly
    initialised model (to be changed after training). User must provide `flops`
    argument to model.

    Args:
        flops: Flops of the model eg. "400MF"
    """

    def __init__(self, flops:str = ""):
        super(RegNetY, self).__init__()

        if flops not in ALLOWED_FLOPS:
            raise ValueError("`flops` must be one of " + str(ALLOWED_FLOPS))
        
        self.flops = flops
        self.config = get_model_config(self.flops)
        self.model = self._get_model_with_config(self.config)

    def build(self, input_shape):
        self.model.build(input_shape)

    def call(self, inputs):
        self.model.call(inputs)
    
    def plot_model(self, imgpath = "model.png"):
        tf.keras.utils.plot_model(
                self.model,
                to_file=imgpath,
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
            )

    def _get_model_with_config(self, config):
        """
        Makes a tf.keras.Sequential model using the given config.
         
        """
        model = tf.keras.models.Sequential()
        model.add(Stem())

        in_channels = 32 # Output channels from Stem

        for i in range(4):  # 4 stages
            depth = config.depths[i]
            out_channels = config.widths[i]
            group_width = config.group_width

            model.add(Stage(
                depth,
                group_width,
                in_channels,
                out_channels
            ))
            
            in_channels = out_channels
        
        model.add(Head(config.num_classes))

        return model


