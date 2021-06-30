import tensorflow as tf
import tensorflow_addons as tfa
import os
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
            applied
        strength: Integer between 0 and 10 specifying level of augmentation
    """
    def __init__(self,
                 num_augs: int = 3,
                 strength: int = 5):
        self.num_augs = num_augs
        self.strength = strength
        self.augs =  self.get_augs(num_augs)

    def color_jitter(self):
        """
        Performs coloer jitter on the batch. 

        Args: None, parameters defined in __init__ are used. 
        """
        pass
    
    @staticmethod
    def aug_mask_mapper():
        pass



    def get_augs(self, num_augs:int):
        """
        Returns a dict mapping augmentations to functions. All augmentations are
        randomized, thus `random_` is omitted from alll function names.
        """
        augs = dict()
        augs[0] = self.color_degrade
        augs[1] = self.color_jitter
        augs[2] = self.cutout
        augs[3] = self.equalize
        augs[4] = self.invert
        augs[5] = self.rotate
        augs[6] = self.sharpen
        augs[7] = self.shear_x
        augs[8] = self.shear_y
        augs[9] = self.solarize
        augs[10] = self.translate_x
        augs[11] = self.translate_y

        return augs


