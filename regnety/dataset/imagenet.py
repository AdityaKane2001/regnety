import math
import tensorflow as tf
import tensorflow_addons as tfa
import os
from datetime import datetime

from typing import Union, Callable, Tuple, List, Type

AUTO = tf.data.AUTOTUNE

_TFRECS_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "filename": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "synset": tf.io.FixedLenFeature([], tf.string),
}
    

class ImageNet:
    """Class for all ImageNet data-related functions, including TFRecord 
    parsing along with augmentation transforms. TFRecords must follow the format
    given in _TFRECS_FORMAT. If not specified otherwise in `augment_fn` argument, following 
    augmentations are applied to the dataset:
    - Color Jitter (random brightness, hue, saturation, contrast, flip)
        This augmentation is inspired by SimCLR (https://arxiv.org/abs/2002.05709). 
        The strength parameter is set to 5, which controlsthe effect of augmentations.
    - Random rotate
    - Random crop and resize

    If `augment_fn` argument is not set to the string "default", it should be set to 
    a callable object. That callable must take exactly two arguments: `image` and `target`
    and must return two values corresponding to the same. 

    If `augment_fn` argument is 'val', then the images will be center cropped to 224x224.

    Args:
       cfg: regnety.regnety.config.config.PreprocessingConfig instance.
    """
    def __init__(
        self,
        cfg
    ):

        self.tfrecs_filepath = cfg.tfrecs_filepath
        self.batch_size = cfg.batch_size
        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size
        self.resize_pre_crop = cfg.resize_pre_crop
        self.augment_fn = cfg.augment_fn
        self.num_classes = cfg.num_classes
        self.cache_dir = cfg.cache_dir
        self.color_jitter = cfg.color_jitter
        self.scale_to_unit = cfg.scale_to_unit
        self.scale_method = cfg.scale_method

        if (self.tfrecs_filepath is None) or (self.tfrecs_filepath == []):
            raise ValueError("List of TFrecords paths cannot be None or empty")
        
        if self.augment_fn == "default":
            self.default_augment = True
            self.val_augment = False
            self.strength = 5
        elif self.augment_fn == "val":
            self.default_augment = False
            self.val_augment = True
            self.strength = -1
        else:
            self.default_augment = False
            self.val_augment = False
            self.strength = -1
        
        
        

    
    def decode_example(self, example_: tf.Tensor) -> dict:
        """Decodes an example to its individual attributes.

        Args:
            example: A TFRecord dataset example.

        Returns:
            Dict containing attributes from a single example. Follows
            the same names as _TFRECS_FORMAT.
        """

        example =  tf.io.parse_example(example_, _TFRECS_FORMAT)
        image = tf.reshape(tf.io.decode_jpeg(
            example["image"]), (self.image_size, self.image_size, 3))
        height = example["height"]
        width = example["width"]
        filename = example["filename"]
        label = example["label"]
        synset = example["synset"]
        return {
            "image": image,
            "height": height,
            "width": width,
            "filename": filename,
            "label": label,
            "synset": synset,
        }


    def _read_tfrecs(self) -> Type[tf.data.Dataset]:
        """Function for reading and loading TFRecords into a tf.data.Dataset.

        Args: None.

        Returns:
            A tf.data.Dataset instance.
        """

        files = tf.data.Dataset.list_files(self.tfrecs_filepath)
        ds = files.interleave(tf.data.TFRecordDataset, 
          num_parallel_calls = AUTO,
          deterministic=False)

        ds = ds.map(
            self.decode_example, 
            num_parallel_calls = AUTO 
        )

        # filename = datetime.now().strftime("%m_%d_%Y_%Hh%Mm")
        # ds = ds.cache(os.path.join(self.cache_dir, filename))
        
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(AUTO)
        return ds
        

    def color_jitter(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Performs color jitter on the batch. It performs random brightness, hue, saturation, 
        contrast and random left-right flip. 

        Args: 
            image: Batch of images to perform color jitter on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """
        
        
        brightness_delta = self.strength * 0.1
        contrast_lower = 1 - 0.5 * (self.strength /10.)
        contrast_upper = 1 + 0.5 * (self.strength /10.)
        hue_delta = self.strength * 0.05
        saturation_lower = 1 - 0.5 * (self.strength /10.)
        saturation_upper = (1 - 0.5 * (self.strength /10.)) * 5
        
        aug_images = tf.image.random_brightness(image, brightness_delta)
        aug_images = tf.image.random_contrast(aug_images, contrast_lower, 
            contrast_upper)
        aug_images = tf.image.random_hue(aug_images, hue_delta)
        aug_images = tf.image.random_saturation(aug_images, saturation_lower, 
            saturation_upper)
        
        
        return aug_images, target

    def random_flip(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Returns randomly flipped batch of images. Only horizontal flip 
        is available

        Args: 
            image: Batch of images to perform random rotation on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """

        aug_images = tf.image.random_flip_left_right(image)
        return aug_images, target


    def random_rotate(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """"
        Returns randomly rotated batch of images.

        Args: 
            image: Batch of images to perform random rotation on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """

        angles = tf.random.uniform((self.batch_size,)) * (math.pi / 2.)
        rotated = tfa.image.rotate(image, angles, fill_value=128.0)
        return rotated, target


    def random_crop(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """"
        Returns random crops of images. 

        Args: 
            image: Batch of images to perform random crop on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """

        cropped = tf.image.random_crop(image, size=(self.batch_size, 320, 320, 3))
        return cropped, target
    
    def center_crop(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Resizes a batch of images to (self.resize_pre_crop, self.resize_pre_crop) and
        then takes central crop of (self.crop_size, self.crop_size)

        Args: 
            image: Batch of images to perform center crop on.
            target: Target tensor.

        Returns:
            Center cropped example with batch of images and targets with same dimensions.
        """
        aug_images = tf.image.resize(
            image, (self.resize_pre_crop, self.resize_pre_crop))
        aug_images = tf.image.central_crop(aug_images, float(self.crop_size)/float(self.resize_pre_crop))
        return aug_images, target

    def _scale_to_unit(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Divides the pixel values of the image by 255. 

        Args: 
            image: Batch of images to perform center crop on.
            target: Target tensor.

        Returns:
            Scaled example with batch of images and targets with same dimensions.
        """
        if self.scale_method == "torch":
            aug_images = tf.cast(image, tf.float32)
            aug_images = aug_images / 127.5
            aug_images = aug_images - 1.
            return aug_images, target
        
        else:
            aug_images = tf.cast(image, tf.float32)
            aug_images = aug_images / 255.
            return aug_images, target
            

    def _one_hot_encode_example(self, example: dict) -> tuple:
        """Takes an example having keys 'image' and 'label' and returns example
        with keys 'image' and 'target'. 'target' is one hot encoded.

        Args:
            example: an example dict having keys 'image' and 'label'.

        Returns:
            Tuple having structure (image_tensor, targets_tensor).
        """
        return (example["image"], tf.one_hot(example["label"], self.num_classes))


    def make_dataset(self) -> Type[tf.data.Dataset]:
        """
        Function to apply all preprocessing and augmentations on dataset using
        tf.data.dataset.map().

        If `augment_fn` argument is not set to the string "default", it should be set to 
        a callable object. That callable must take exactly two arguments: `image` and `target`
        and must return two values corresponding to the same. 

        Args: None.

        Returns:
            tf.data.Dataset instance having the final format as follows:
            (image, target)
        """
        ds = self._read_tfrecs()
        
        
        ds = ds.map(
            self._one_hot_encode_example,
            num_parallel_calls=AUTO
        )


        if self.default_augment:
            if self.color_jitter:
                ds = ds.map(
                    self.color_jitter,
                    num_parallel_calls=AUTO
                )
            
            ds = ds.map(
                self.random_flip,
                num_parallel_calls=AUTO
            )

            ds = ds.map(
                self.random_rotate,
                num_parallel_calls=AUTO
            )

            ds =  ds.map(
                self.random_crop,
                num_parallel_calls=AUTO
            )
        
        elif self.val_augment:
            ds = ds.map(
                self.center_crop,
                num_parallel_calls=AUTO
            )

        else:
            ds = ds.map(
                self.augment_fn,
                num_parallel_calls=AUTO
            )
        
        
        if self.scale_to_unit:
            ds = ds.map(
                self._scale_to_unit,
                num_parallel_calls=AUTO
            )



        return ds
