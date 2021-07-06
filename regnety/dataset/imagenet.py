import tensorflow as tf
import os

from regnety.regnety.dataset.augment import WeakRandAugment
from official.vision.image_classification.augment import RandAugment
from typing import Union, Callable, Tuple, List, Type

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
    given below. If not specified otherwise in `augment_fn` argument, following 
    augmentations are applied to the dataset:
    - Random sized crop (train only)
    - Scale and center crop (validation and test only)
    Args:
        tfrecs_filepath: list of filepaths of all TFRecords files
        batch_size: batch_size for the Dataset
        image_size: final image size of the images in the dataset
        augment_fn: function to apply to dataset after loading raw TFrecords
        num_classes: number of classes
        randaugment: True if RandAugment is to be applied after other 
            preprocessing functions. It is applied even if this is True and 
            augment_fn is not 'default'. It is not applied in any case if this
            argument is False.
    """
    def __init__(
        self,
        tfrecs_filepath: List[str]  = None,
        batch_size: int = 1024,
        image_size: int = 512,
        augment_fn: Union[str, Callable]  = "default",
        num_classes: int = 10,
        randaugment: bool = True,
    ):

        if (tfrecs_filepath is None) or  (tfrecs_filepath == []):
            raise ValueError("List of TFrecords paths cannot be None or empty")
        self.tfrecs_filepath = tfrecs_filepath
        self.batch_size = batch_size
        self.image_size = tf.constant([image_size] * self.batch_size) 
        self.augment_fn = augment_fn
        self.num_classes = num_classes
        self.randaugment = randaugment
        if self.randaugment:
            self._augmenter = WeakRandAugment(strength=5, num_augs=2, batch_size = batch_size)
            self.strength = 5


    
    def decode_example(self, example_: tf.Tensor) -> dict:
        """Decodes an example to its individual attributes
        Args:
            example: A TFRecord dataset example.
        Returns:
            Dict containing attributes from a single example. Follows
            the same names as TFRECORDS_FORMAT.
        """

        example =  tf.io.parse_example(example_, _TFRECS_FORMAT)
        image = tf.reshape(tf.io.decode_jpeg(example["image"]), (512,512,3))
        height = example['height']
        width = example['width']
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
        Returns:
            A tf.data.Dataset
        """

        files = tf.data.Dataset.list_files(self.tfrecs_filepath)


        #files = files.take(1)
        options = tf.data.Options()


        options.experimental_deterministic = False

        files = files.with_options(options)

        ds = files.interleave(tf.data.TFRecordDataset, 
          num_parallel_calls = tf.data.AUTOTUNE,
          deterministic=False)

       
        ds = ds.map(
            self.decode_example, 
            num_parallel_calls = tf.data.AUTOTUNE 
        )

        #ds = ds.cache()
       
        ds = ds.repeat()
        ds = ds.batch(self.batch_size)
        #ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


    
    def _one_hot_encode_example(self, example: dict) -> dict:
        """Takes an example having keys 'image' and 'label' and returns example
        with keys 'image' and 'target'. 'target' is one hot encoded.
        Args:
            example: an example having keys 'image' and 'label'
        Returns:
            example having keys 'image' and 'target'
        """
        return (example["image"], tf.one_hot(example["label"], self.num_classes))

    def color_jitter(self, example: dict) -> dict:
        """
        Performs color jitter on the batch. Random brightness, hue, saturation, 
        and contrast

        Args: 
            images: batch of images to be augmented

        Returns:
            Augmented batch of images with same dimensions 
        """
        
        images = example['image']
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
        
        return {
            "image": aug_images,
            "height": self.image_size,
            "width": self.image_size,
            "filename": example["filename"],
            "label": example["label"],
            "synset": example["synset"],
        }


    def _randaugment(self, example: dict) -> dict:
        """Wrapper for tf vision's RandAugment.distort function which
        accepts examples as input instead of images. Uses magnitude = 5
        as per pycls/pycls/datasets/augment.py#L29.
        Args:
            example: Example having the key 'image'
        Returns:
            example in which RandAugment has been applied to the image
        """
        image = example['image']

        image = self._augmenter.apply_augs(image)
        return {
            "image": image,
            "height": self.image_size,
            "width": self.image_size,
            "filename": example["filename"],
            "label": example["label"],
            "synset": example["synset"],
        }


    def make_dataset(self) -> Type[tf.data.Dataset]:
        """
        Function to apply all preprocessing and augmentations on dataset using
        dataset.map().
        Returns:
            Dataset having the final format as follows:
            {
                'image' : (batch_size, self.image_size, self.image_size, 3)
                'target' : (num_classes,)
            }
        """
        ds = self._read_tfrecs()
        
        

        if self.randaugment:
            ds = ds.map(
                self.color_jitter,
                num_parallel_calls = tf.data.AUTOTUNE
            )
        
        ds = ds.map(
            self._one_hot_encode_example,
            num_parallel_calls = tf.data.AUTOTUNE
        )
        
           


        return ds