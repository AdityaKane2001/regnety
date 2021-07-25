import tensorflow as tf

from dataclasses import dataclass
from typing import List, Type, Union, Callable

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
        optimizer: One of "sgd", "adam", "adamw"
        base_lr: Base learning rate for training
        warmup_epochs: Number of epochs used for warmup
        warmup_factor: Gradual linear warmup factor
        total_epochs: Number of training epochs
        weight_decay: Weight decay to be used in optimizer
        momentum: Momentum to be used in optimizer
        lr_schedule: One of "constant" or "half_cos"
        log_dir: Path to store logs
        model_dir: Path to store model checkpoints
    """

    optimizer: str
    base_lr: float
    warmup_epochs: int
    warmup_factor: float
    total_epochs: int
    weight_decay: float
    momentum: float
    lr_schedule: str
    log_dir: str
    model_dir: str


@dataclass
class PreprocessingConfig:
    tfrecs_filepath: List[str]
    batch_size: int
    image_size: int
    crop_size: int
    resize_pre_crop: int
    augment_fn: Union[str, Callable]
    num_classes: int
    color_jitter: bool
    scale_to_unit: bool
    scale_method: str


def get_preprocessing_config(
    tfrecs_filepath: List[str] = None,
    batch_size: int = 1024,
    image_size: int = 512,
    crop_size: int = 224,
    resize_pre_crop: int = 320,
    augment_fn: Union[str, Callable] = "default",
    num_classes: int = 1000,
    color_jitter: bool = False,
    scale_to_unit: bool = True,
    scale_method: str = "tf"
):

    """
    Getter function for preprocessing configuration. Images are first resized to `resize_pre_crop`
    and then a central crop of `crop_size` is taken.  

    Args:
        tfrecs_filepath: list of filepaths of all TFRecords files.
        batch_size: batch_size for the Dataset.
        image_size: final image size of the images in the dataset.
        crop_size: size to take crop of
        resize_pre_crop: size to resize to before cropping
        augment_fn: function to apply to dataset after loading raw TFrecords.
        num_classes: number of classes.
        color_jitter: If True, color_jitter augmentation is applied.
        scale_to_unit: Whether the images should be scaled using `scale_method`.
        scale_method: Use `tf` if images must be in [0,1]. Use `torch` if images must
            be in [-1,1]
    
    Returns:
        A PreprocessingConfig dataclass instance with all attributes
    
    """
    return PreprocessingConfig(
        tfrecs_filepath=tfrecs_filepath,
        batch_size=batch_size,
        image_size=image_size,
        crop_size=crop_size,
        resize_pre_crop=resize_pre_crop,
        augment_fn=augment_fn,
        num_classes=num_classes,
        color_jitter=color_jitter,
        scale_to_unit=scale_to_unit,
        scale_method=scale_method
    )


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

def get_train_config(
    optimizer: str = "adamw",
    base_lr: float = 0.001 * 8,
    warmup_epochs: int = 5,
    warmup_factor: float = 0.1,
    total_epochs: int = 100,
    weight_decay: float = 5e-5,
    momentum: float = 0.9,
    lr_schedule: str = "half_cos",
    log_dir: str = "gs://ak-europe-train/logs",
    model_dir: str = "gs://ak-europe-train/models",
):
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
        optimizer=optimizer,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        warmup_factor=warmup_factor,
        total_epochs=total_epochs,
        weight_decay=weight_decay,
        momentum=momentum,
        lr_schedule=lr_schedule,
        log_dir=log_dir,
        model_dir=model_dir,
    )
