import tensorflow as tf

from dataclasses import dataclass
from typing import List, Type

ALLOWED_FLOPS = ('200mf', '400mf', '600mf', '800mf')


@dataclass
class RegNetYConfig:
    """
    Dataclass for architecture configuration for RegNetY.

    Args:
        name: Name of the model eg. "RegNetY200MF"
        flops: Flops of the model eg. "400MF" (Processing one image requires 
            400 million floating point operations (multiplication, addition))
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
    num_classes: int
    depths: List[int]
    widths: List[int]
    group_width: int
    bottleneck_ratio: int
    SE_ratio: float
    wa: int
    w0: int
    wm: float


@dataclass
class TrainConfig:
    """
    Dataclass of training configuration for RegNetY

    Args:
        optimizer_strategy: One of "sgd", "adam", "adamw"
        base_lr: Base learning rate for training
        warmup_epochs: Number of epochs used for warmup
        total_epochs: Number of training epochs
        weight_decay: Weight decay to be used in optimizer
        momentum: Momentum to be used in optimizer
        lr_schedule: One of "constant" or "half_cos"
        log_dir: Path to store logs
        model_dir: Path to store model checkpoints
    """

    optimizer_strategy: str
    base_lr: float
    warmup_epochs: int
    total_epochs: int
    weight_decay: float
    momentum: float
    lr_schedule: str
    log_dir: str
    model_dir: str


def get_model_config(flops: str):
    """
    Getter function for configuration for a specific RegNetY model instance.
    User must provide flops in string format. Example "200MF", "800MF".

    Link to the paper: https://arxiv.org/pdf/2003.13678.pdf

    The widths and depths are deduced from a quantized linear function. For 
    more information, please refer to the original paper. 

    Args: 
        flops: String denoting flops of model (as per the paper)

    Returns:
        A RegNetYConfig dataclass instance with all attributes.
    """

    if flops == "":
        raise ValueError("Please enter `flops` argument.")

    if flops.lower() not in ALLOWED_FLOPS:
        raise ValueError("`flops` must be one of " + str(ALLOWED_FLOPS))

    if flops.lower() == '200mf':
        return RegNetYConfig(
            name="RegNetY 200MF",
            flops="200MF",
            num_classes=1000,
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            bottleneck_ratio=1,
            SE_ratio=0.25,
            wa=36,
            w0=24,
            wm=2.5
        )

    elif flops.lower() == '400mf':
        return RegNetYConfig(
            name="RegNetY 400MF",
            flops="400MF",
            num_classes=1000,
            depths=[1, 3, 3, 6],
            widths=[48, 104, 208, 440],
            group_width=8,
            bottleneck_ratio=1,
            SE_ratio=0.25,
            wa=28,
            w0=48,
            wm=2.1
        )

    elif flops.lower() == '600mf':
        return RegNetYConfig(
            name="RegNetY 600MF",
            flops="600MF",
            num_classes=1000,
            depths=[1, 3, 7, 4],
            widths=[48, 112, 256, 608],
            group_width=16,
            bottleneck_ratio=1,
            SE_ratio=0.25,
            wa=33,
            w0=48,
            wm=2.3
        )

    elif flops.lower() == '800mf':
        return RegNetYConfig(
            name="RegNetY 800MF",
            flops="800MF",
            num_classes=1000,
            depths=[1, 3, 8, 2],
            widths=[64, 128, 320, 768],
            group_width=16,
            bottleneck_ratio=1,
            SE_ratio=0.25,
            wa=39,
            w0=56,
            wm=2.4
        )


def get_train_config():
    """
    Getter function for training config. Config is same as in the paper
    (see link above). If ambiguous, 
    https://github.com/facebookresearch/pycls/blob/master/pycls/core/config.py  
    is assumed to be source of truth.

    Args: None

    Returns:
        A TrainConfig dataclass instance.
    """

    return TrainConfig(
        optimizer_strategy="sgd",
        base_lr=0.1,
        warmup_epochs=5,
        total_epochs=100,
        weight_decay=5e-4,
        momentum=0.9,
        lr_schedule="half_cos",
        log_dir="gs://adityakane-train/logs",
        model_dir="gs://adityakane-train/models"
    )
