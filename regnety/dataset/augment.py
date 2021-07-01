import os
import tensorflow as tf
import tensorflow_addons as tfa

from typing import Union, Callable, Tuple, List, Type

class WeakRandAugment:
    """
    Implements a weaker version of RandAugment. Is vectorized. 
    0: Color Degradation: Degrades image
    1: Color Jitter: Random brightness, hue, saturation and contrast
    2: Cutout : cutout,
    3: Equalize: equalize image ,
    4: Invert: invert image randomly,
    5: Rotate: Rotate image randomly,
    6: Sharpness: sharpness,
    7: ShearX : shear_x,
    8: ShearY : shear_y,
    9: Solarize: 
    10: TranslateX : translate_x,
    11: TranslateY : translate_y,
    

    Args:
        num_augs: Integer specifying number of random augmentations to be 
            applied. Must be >= 1 and <=3 
        strength: Integer between 0 and 10 specifying level of augmentation
    """
    def __init__(self,
                 num_augs: int = 2,
                 strength: int = 5,
                 batch_size: int = 128):

        if num_augs < 1 or num_augs > 3:
            raise ValueError("`num_augs` must be between 1 and 3")
        self.num_augs = tf.constant(num_augs, dtype = tf.int32)
        self.strength = tf.constant(strength, dtype = tf.float32)
        self.batch_size = batch_size
        self.augs =  self._get_augs(num_augs)


    def blend(image1:tf.Tensor, image2:tf.Tensor, factor:tf.Tensor) -> tf.Tensor:
        """
        Blends two batches of images by a factor. Factor must be a
        Tensor of shape (batch_size,1,1,1)

        Args:
            image1: One of the images to be mixed
            image2: One of the images to be mixed
            factor: This amount of image1 will be taken in the mix. 1-factor
                will be amount image2 will be taken in the mix.
        
        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)
        """
        if tf.rank(factor) < 4:
            raise ValueError(
                "Rank of `factor` is less than 4. Factor must have the shape (batch_size, 1, 1, 1)")
        
        aug_image = image1 * factor + image2 * (1. - factor)

        return aug_image


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
        contrast_lower = 1 - 0.5 * self.strength
        contrast_upper = 1 + 0.5 * self.strength
        hue_delta = self.strength * 0.05
        saturation_lower = 1 - 0.5 * self.strength
        saturation_upper = (1 - 0.5 * self.strength) * 5
        
        aug_images = tf.image.random_brightness(images, brightness_delta)
        aug_images = tf.image.random_contrast(aug_images, contrast_lower, 
            contrast_upper)
        aug_images = tf.image.random_hue(aug_images, hue_delta)
        aug_images = tf.image.random_saturation(aug_images, saturation_lower, 
            saturation_upper)

        return aug_images
    

    def color_degradation(self, images:tf.Tensor) -> tf.Tensor:
        """
        Converts RGB to grayscale and back.

        Args:
            images: Batch of images

        Returns:
           Tensor of shape (batch_size, image_size, image_size, channels)
        """

        factor = tf.random.uniform((self.batch_size,1,1,1))
        degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
        return self.blend(degenerate, image, factor)


    def cutout(self, images: tf.Tensor) -> tf.Tensor:
        """
        Applies random cutout to images. 
        
        Args:
            images: batch of images to apply cutout to

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)        
        """
        mask_size = (224 * self.strength) / 100.
        mask_size = tf.math.round(mask_size)
        if mask_size % 2 != 0:
            mask_size += 1
        
        aug_images = tfa.image.random_cutout(images, mask_size, constant_values = 128.)
        return aug_images
    

    def equalize(self, images:tf.Tensor) -> tf.Tensor:
        """
        Applies equalize augmentation to images

        Args:
            images: batch of images to apply equalize to

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)
        """

        return tfa.image.equalize(images, bins = 256)


    def invert(self, images:tf.Tensor) -> tf.Tensor:
        """
        Inverts given images

        Args:
            images: batch of images to invert

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)
        """
        return 255. - images

    def rotate(self, images: tf.tenosr) -> tf.Tensor:
        """
        Randomly rotates given images

        Args:
            images: batch of images to randomly rotate

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)
        """
        
        PI = tf.constant(3.141592653589793)
        angles = tf.random.uniform((self.batch_size)) * PI/2
        return tfa.rotate(images, angles, fill_value = tf.constant([128.]))


    def sharpen(self, images:tf.Tensor) -> tf.Tensor:
        """
        Sharpens the images by a random factor

        Args:
            images: batch of images to randomly rotate

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)
        """

        factor = tf.random.uniform((),minval = 0.01, maxval = 1) * self.strength
        return tfa.image.sharpness(images, factor)
    

    def shear_x(self, images: tf.Tensor) -> tf.Tensor:
        """
        Randomly shears the images in X direction

        Args:
            images: batch of images to randomly shear in X direction

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)
        
        """
        level = self.strength / 10.
        replace = tf.constant([128.])
        return tfa.image.shear_x(images, level, replace)
    

    def shear_y(self, images: tf.Tensor) -> tf.Tensor:
        """
        Randomly shears the images in Y direction

        Args:
            images: batch of images to randomly shear in Y direction

        Returns:
            Tensor of shape (batch_size, image_size, image_size, channels)
        
        """
        level = self.strength / 10.
        replace = tf.constant([128.])
        return tfa.image.shear_y(images, level, replace)


    def _get_augs(self) -> tf.Tensor :
        """
        Randomly selects num_augs augmentations for the batch
        
        Args: 
            augs_list: List of augmentations from get_augs_list function
        
        Returns:
            A list having num_augs entries having integers representing 
            augmentations to be applied.
        """
        augs = tf.random.uniform(shape = (self.num_augs,),
            minval = 0, maxval = 12, dtype = tf.int32)
        augs = tf.sort(augs)
        return augs

    def apply_augs(self, images: tf.Tensor) -> tf.Tensor:
        """
        Applies augmentations denoted by `get_augs`.

        Args: 
            images: Batch of images with shape (None, 224, 224, 3)

        Returns:
            Augmented batch of images of size (None, 224, 224, 3)
        """
        
        aug_images = tf.cast(images, tf.float32)

        for i in self.augs:
            if i == 0:
                aug_images = self.color_degrade(aug_images)
            
            if i == 1:
                aug_images = self.color_jitter(aug_images)
            
            if i == 2:
                aug_images = self.cutout(aug_images)
            
            if i == 3:
                aug_images = self.equalize(aug_images)
            
            if i == 4:
                aug_images = self.invert(aug_images)
            
            if i == 5:
                aug_images = self.rotate(aug_images)
            
            if i == 6:
                aug_images = self.sharpen(aug_images)
            
            if i == 7:
                aug_images = self.shear_x(aug_images)
            
            if i == 8:
                aug_images = self.shear_y(aug_images)
            
            if i == 9:
                aug_images = self.solarize(aug_images)
            
            if i == 10:
                aug_images = self.translate_x(aug_images)
            
            if i == 11:
                aug_images = self.translate_y(aug_images)
            
        return tf.cast(aug_images, tf.uint8)


    # def get_aug_from_index(self, aug_index:int) -> Callable:
    #     """
    #     Returns a callable given augment index. All augmentations are
    #     randomized, thus `random_` is omitted from alll function names.

    #     Args: 

    #     """
    #     augs = []
    #     augs.append(self.color_degrade)  # augs[0] = self.color_degrade
    #     augs.append(self.color_jitter)   # augs[1] = self.color_jitter
    #     augs.append(self.cutout)         # augs[2] = self.cutout
    #     augs.append(self.equalize)       # augs[3] = self.equalize
    #     augs.append(self.invert)         # augs[4] = self.invert
    #     augs.append(self.rotate)         # augs[5] = self.rotate
    #     augs.append(self.sharpen)        # augs[6] = self.sharpen
    #     augs.append(self.shear_x)        # augs[7] = self.shear_x
    #     augs.append(self.shear_y)        # augs[8] = self.shear_y
    #     augs.append(self.solarize)       # augs[9] = self.solarize
    #     augs.append(self.translate_x)    # augs[10] = self.translate_x
    #     augs.append(self.translate_y)    # augs[11] = self.translate_y

    #     return augs


