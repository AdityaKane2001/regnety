import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa

from typing import Union, Callable, Tuple, List, Type

class WeakRandAugment:
    """
    Implements a weaker version of RandAugment. Is vectorized. 
    0: Color Jitter: Random brightness, hue, saturation and contrast
    1: Cutout : cutout,
    2: Invert: invert image randomly,
    3: Rotate: Rotate image randomly,
    4: Solarize: Invert pixels less than threshold

    

    Args:
        num_augs: Integer specifying number of random augmentations to be 
            applied. Must be >= 1 and <=3 
        strength: Integer between 0 and 10 specifying level of augmentation
    """
    def __init__(self,
                 num_augs: int = 2,
                 strength: float = 5.,
                 batch_size: int = 128):

        if num_augs < 1 or num_augs > 3:
            raise ValueError("`num_augs` must be between 1 and 3")
        self.num_augs = tf.constant(num_augs, dtype = tf.int32)
        self.strength = strength
        self.batch_size = batch_size
        self.augs =  self._get_augs()


   
    def color_jitter(self, images: tf.Tensor) -> tf.Tensor:
        """
        Performs color jitter on the batch. Random brightness, hue, saturation, 
        and contrast

        Args: 
            images: batch of images to be augmented

        Returns:
            Augmented batch of images with same dimensions 
        """
        
        brightness_delta = self.strength * 0.1
        contrast_lower = 1 - 0.5 * (self.strength /10.)
        contrast_upper = 1 + 0.5 * (self.strength /10.)
        hue_delta = self.strength * 0.05
        saturation_lower = 1 - 0.5 * (self.strength /10.)
        saturation_upper = (1 - 0.5 * (self.strength /10.)) * 5
        
        aug_images = tf.image.random_brightness(images, brightness_delta)
        aug_images = tf.image.random_contrast(aug_images, contrast_lower, 
            contrast_upper)
        aug_images = tf.image.random_hue(aug_images, hue_delta)
        aug_images = tf.image.random_saturation(aug_images, saturation_lower, 
            saturation_upper)
        
        return aug_images
    

    def cutout(self, images: tf.Tensor) -> tf.Tensor:
        """
        Applies random cutout to images. 
        
        Args:
            images: batch of images to apply cutout to

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)        
        """
        mask_size = (224 * self.strength) / 10.
        mask_size = tf.cast(tf.math.ceil(mask_size), tf.int32)
        
        mask_size = tf.cond(mask_size % 2 == 0,lambda: mask_size,lambda:mask_size + 1)
        
        aug_images = tfa.image.random_cutout(images, mask_size, constant_values = 128.)
        return aug_images
    

    def equalize(self, images: tf.Tensor) -> tf.Tensor:
        """
        Equalizes given batch of images
        
        Args:
            images: batch of images to equalize

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)
        """
        return tfa.image.equalize(images)


    def invert(self, images:tf.Tensor) -> tf.Tensor:
        """
        Inverts given images

        Args:
            images: batch of images to invert

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)
        """
        
        return tf.cast(255 - images, tf.uint8)

    def rotate(self, images: tf.Tensor) -> tf.Tensor:
        """
        Randomly rotates given images

        Args:
            images: batch of images to randomly rotate

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)
        """
        
        PI = tf.constant(3.141592653589793)
        angles = tf.random.uniform(()) * PI
        return tfa.image.rotate(images, angles, fill_value = 128.)


    def sharpen(self, images: tf.Tensor) -> tf.Tensor:
        """
        Sharpens the image based on given strength
        Args:
            images: batch of images to sharpen

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)
        """
        factor = self.strength 
        return tfa.image.sharpness(images, factor)  

    
    def solarize(self, images: tf.Tensor) -> tf.Tensor:
        """
        Solarizes images and blends then randomly with original images
        
        Args:
            images: batch of images to solarize

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)
        """

        solarized = tf.where(images < 128, images, 255 - images)

        return solarized


    def _get_augs(self) -> tf.Tensor :
        """
        Randomly selects num_augs augmentations for the batch
        
        Args: 
            augs_list: List of augmentations from get_augs_list function
        
        Returns:
            A list having num_augs entries having integers representing 
            augmentations to be applied.
        """
        augs = tf.random.uniform((self.num_augs,), minval = 0, maxval = 4, dtype = tf.int32)
        augs = tf.sort(augs)
        return tf.data.Dataset.from_tensor_slices(augs)


    def apply_augs(self, images: tf.Tensor) -> tf.Tensor:
        """
        Applies augmentations denoted by `get_augs`.

        Args: 
            images: Batch of images with shape (None, 224, 224, 3)

        Returns:
            Augmented batch of images of size (None, 224, 224, 3)
        """
        
        aug_images = tf.cast(images, tf.uint8)
        aug_functions = self.get_aug_list(images)
        

        for i in self.augs:
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(aug_images, 
                    tf.TensorShape([self.batch_size, 512, 512, 3]))])

            aug_images = tf.switch_case(
                i,
                aug_functions
            )

            
        return tf.cast(aug_images, tf.uint8)


    def get_aug_list(self, batch) -> dict:
        """
        Returns a callable given augment index. All augmentations are
        randomized, thus `random_` is omitted from alll function names.

        Args: 

        """
        augs = dict()
        
        augs[0] = lambda: self.color_jitter(batch)
        augs[1] = lambda: self.cutout(batch)
        augs[2] = lambda: self.equalize(batch)
        augs[3] = lambda: self.invert(batch)
        augs[4] = lambda: self.rotate(batch)
        augs[5] = lambda: self.sharpen(batch)
        augs[6] = lambda: self.solarize(batch)
        

        return augs
 