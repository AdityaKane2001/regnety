import tensorflow as tf
import os
from dataclasses import dataclass
from official.vision.image_classification.augment import RandAugment

_TFRECS_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "filename": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "synset": tf.io.FixedLenFeature([], tf.string),
}
    

class ImageNet:
    """Class for all ImageNet related functions, includes TFRecords loading,
    preprocessing and augmentations. TFRecords must follow the format given
    below. If not specified otherwise in `augment_fn` argument, following 
    augmentations are applied to the dataset:
    - Random sized crop (train only)
    - Scale and center crop

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
        tfrecs_filepath: str = None,
        batch_size: int = 128,
        image_size: int = 224,
        augment_fn: str = "default",
        num_classes: int = 10,
        randaugment: bool = True,
    ):

        if tfrecs_filepath is None:
            raise ValueError("List of TFrecords paths cannot be None")
        self.tfrecs_filepath = tfrecs_filepath
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment_fn = augment_fn
        self.num_classes = num_classes
        self.randaugment = randaugment
        if self.randaugment:
            self._augmenter = RandAugment(magnitude=5, num_layers=2)

    def decode_example(self, example: tf.Tensor):
        """Decodes an example to its individual attributes

        Args:
            example: A TFRecord dataset example.

        Returns:
            Dict containing attributes from a single example. Follows
            the same names as TFRECORDS_FORMAT.
        """
        image = tf.cast(tf.io.decode_jpeg(example["image"]), tf.float32)
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

    def _read_tfrecs(self) -> tf.data.Dataset:
        """Function for reading and loading TFRecords into a tf.data.Dataset.

        Returns:
            A tf.data.Dataset
        """
        ds = tf.data.TFRecordDataset(self.tfrecs_filepath)
        ds = ds.map(
            lambda example: tf.io.parse_example(example, _TFRECS_FORMAT)
        )
        ds = ds.map(lambda example: self.decode_example(example))
        return ds

    def _scale_and_center_crop(self, image, scale_size, final_size):
        """Resizes image to given scale size and returns a center crop. Aspect
        ratio is maintained. Note that final_size must be less than or equal to
        scale_size.
        Args:
            image: tensor of the image
            scale_size: Size of image to scale to
            final_size: Size of final image

        Returns:
            Tensor of shape (final_size, final_size, 3)
        """
        if final_size > scale_size:
            raise ValueError('final_size must be greater than scale_size, recieved %d and %d respectively' % (final_size, scale_size))

        square_scaled_image = tf.image.resize_with_pad(image, 
            scale_size, scale_size) 
        return tf.image.central_crop(
            tf.cast(final_size, tf.float32) / scale_size)
        

    def random_sized_crop(self, example, min_area=0.08):
        """
        Takes a random crop of image. Resizes it to self.image_size. Aspect 
        ratio is NOT maintained. 
        Code inspired by: https://tinyurl.com/tekd7yaz

        Args:
            example: A dataset example.
            min_area: Minimum area of image to be used

        Returns:
            Example of same format as _TFRECS_FORMAT
        """

        image = tf.cast(example['image'], tf.float32)
        h = tf.cast(example['height'], tf.int64)
        w = tf.cast(example['width'], tf.int64)


        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], 
                         dtype=tf.float32,
                         shape=[1, 1, 4])
        

        crop_begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
           [h, w, 3],
            bbox,
            min_object_covered = 0.08,
            area_range = [0.08, 1.0],
            max_attempts = 10
        )

        distorted_image = tf.slice(image, crop_begin, crop_size)

        image = tf.image.resize(distorted_image, 
            (self.image_size, self.image_size))

        
        return {
            "image": image,
            "height": self.image_size,
            "width": self.image_size,
            "filename": example["filename"],
            "label": example["label"],
            "synset": example["synset"],
        }

    def _one_hot_encode_example(self, example):
        """Takes an example having keys 'image' and 'label' and returns example
        with keys 'image' and 'target'. 'target' is one hot encoded.

        Args:
            example: an example having keys 'image' and 'label'

        Returns:
            example having keys 'image' and 'target'
        """
        return {
            "image": example["image"],
            "target": tf.one_hot(example["label"], self.num_classes),
        }

    def _randaugment(self, example):
        """Wrapper for tf vision's RandAugment.distort function which
        accepts examples as input instead of images. Uses magnitude = 5
        as per pycls/pycls/datasets/augment.py#L29.

        Args:
            example: Example having the key 'image'

        Returns:
            example in which RandAugment has been applied to the image
        """
        example['image'] = self._augmenter.distort(example['image'])
        return example

    def make_dataset(
        self,
    ):
        """
        Function to apply all preprocessing and augmentations on dataset using
        dataset.map().

        Returns:
            Dataset having the final format as follows:
            {
                'image' : (self.image_size, self.image_size, 3)
                'target' : (num_classes,)
            }
        """
        ds = self._read_tfrecs()

        if self.augment_fn == "default":
            ds = ds.map(lambda example: self.random_sized_crop(example))
            

        else:
            ds = ds.map(lambda example: self.augment_fn(example))

        if self.randaugment:
                ds = ds.map(lambda example: self._randaugment(example))
        
        #ds = ds.map(lambda example: self._one_hot_encode_example(example))

        return ds
