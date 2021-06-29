from dataclasses import dataclass
<<<<<<< HEAD
from typing import List

ALLOWED_FLOPS = ('200mf', '400mf', '600mf', '800mf')

@dataclass
class RegNetYConfig:
    """
    Dataclass for architecture configuration for RegNetY.

=======
from typing import List,

@dataclass
class RegNetYConfig:
    """
    Dataclass for architecture configuration for RegNetY.

>>>>>>> e4914bf (Added config.py)
    Args:
        name: Name of the model eg. "RegNetY200MF"
        flops: Flops of the model eg. "400MF"
        depths: List of depths for every stage
        widths: List of  widths (number of channels) after every stage
        group_width: Integer denoting groups in every convolution layer
        bottleneck_ratio: Integer specifying bottleneck ratio
        SE_ratio: Float denoting squeeze and excite ratio
        wa: Integer, slope used in linear parameterization
        w0: Integer, inital value used in linear parameterization
        wm: Float, quantization parameter
    """

    name: str
    flops: str
<<<<<<< HEAD
    num_classes: int
=======
>>>>>>> e4914bf (Added config.py)
    depths: List[int]
    widths: List[int]
    group_width: int
    bottleneck_ratio: int
    SE_ratio: float
    wa: int
    w0: int
    wm: float



<<<<<<< HEAD
def get_model_config(flops: str):
=======
def get_config(flops: str):
>>>>>>> e4914bf (Added config.py)
    """
    Getter function for configuration for a specific RegNetY model instance.
    User must provide flops in string format. Example "200MF", "800MF" 
    
    Args: 
        flops: String denoting flops of model (as per the paper)

    Returns:
        A RegNetYConfig dataclass instance with all attributes.
    """
    
    if flops == "":
        raise ValueError("Please enter `flops` argument.")
    
<<<<<<< HEAD
    

    if flops.lower() not in ALLOWED_FLOPS:
        raise ValueError("`flops` must be one of " + str(ALLOWED_FLOPS))
=======
    allowed_flops = ['200mf', '400mf', '600mf', '800mf']

    if flops.lower() not in allowed_flops:
        raise ValueError("`flops` must be one of " + str(allowed_flops))
>>>>>>> e4914bf (Added config.py)
    
    if flops.lower() == '200mf':
        return RegNetYConfig(
            name = "RegNetY 200MF",
            flops = "200MF",
<<<<<<< HEAD
            num_classes = 1000,
=======
>>>>>>> e4914bf (Added config.py)
            depths = [1, 1, 4, 7],
            widths = [24, 56, 152, 368],
            group_width = 8,
            bottleneck_ratio = 1,
            SE_ratio = 0.25,
            wa = 36,
            w0 = 24,
            wm = 2.5
        )
        
    elif flops.lower() == '400mf':
        return RegNetYConfig(
            name = "RegNetY 400MF",
            flops = "400MF",
<<<<<<< HEAD
            num_classes = 1000,
=======
>>>>>>> e4914bf (Added config.py)
            depths = [1, 3, 3, 6],
            widths = [48, 104, 208, 440],
            group_width = 8,
            bottleneck_ratio = 1,
            SE_ratio = 0.25,
            wa = 28,
            w0 = 48,
            wm = 2.1
        )
    
    elif flops.lower() == '600mf':
        return RegNetYConfig(
            name = "RegNetY 600MF",
            flops = "600MF",
<<<<<<< HEAD
            num_classes = 1000,
=======
>>>>>>> e4914bf (Added config.py)
            depths = [1, 3, 7, 4],
            widths = [48, 112, 256, 608],
            group_width = 16,
            bottleneck_ratio = 1,
            SE_ratio = 0.25,
            wa = 33,
            w0 = 48,
            wm = 2.3
        )
    
    elif flops.lower() == '800mf':
        return RegNetYConfig(
            name = "RegNetY 800MF",
            flops = "800MF",
<<<<<<< HEAD
            num_classes = 1000,
=======
>>>>>>> e4914bf (Added config.py)
            depths = [1, 3, 8, 2],
            widths = [64, 128, 320, 768],
            group_width = 16,
            bottleneck_ratio = 1,
            SE_ratio = 0.25,
            wa = 39,
            w0 = 56,
            wm = 2.4
        )
<<<<<<< HEAD
=======
   
>>>>>>> e4914bf (Added config.py)
