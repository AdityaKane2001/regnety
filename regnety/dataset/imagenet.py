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
        batch_size: int = 128,
        image_size: int = 224,
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

    @tf.function
    def decode_example(self, example: tf.Tensor) -> dict:
        """Decodes an example to its individual attributes
        Args:
            example: A TFRecord dataset example.
        Returns:
            Dict containing attributes from a single example. Follows
            the same names as TFRECORDS_FORMAT.
        """
        image = tf.io.decode_jpeg(example["image"])
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
            lambda example: tf.io.parse_example(example, _TFRECS_FORMAT),
            num_parallel_calls = tf.data.AUTOTUNE
        )
        ds = ds.map(
            lambda example: self.decode_example(example), 
            num_parallel_calls = tf.data.AUTOTUNE 
        )

        ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.repeat()
        ds = ds.batch(self.batch_size)
        return ds

    # def _scale_and_center_crop(self, 
    #     image: tf.Tensor,
    #     scale_size: tf.Tensor, 
    #     final_size: tf.Tensor) -> tf.Tensor:
    #     """Resizes image to given scale size and returns a center crop. Aspect
    #     ratio is maintained. Note that final_size must be less than or equal to
    #     scale_size.
    #     Args:
    #         image: tensor of the image
    #         scale_size: Size of image to scale to
    #         final_size: Size of final image
    #     Returns:
    #         Tensor of shape (final_size, final_size, 3)
    #     """
    #     if final_size < scale_size:
    #         raise ValueError('final_size must be lesser than scale_size, recieved %d and %d respectively' % (final_size, scale_size))

    #     square_scaled_image = tf.image.resize_with_pad(image, 
    #         scale_size, scale_size) 
    #     return tf.image.central_crop(square_scaled_image, 
    #         final_size / scale_size)
    
    # def _get_boxes(self, aspect_ratio, area):
    #     """
    #     Returns crop boxes to be used in crop_and_resize
    #     """
    #     heights = tf.random.uniform((self.batch_size,), 
    #         maxval = tf.math.sqrt(area) * aspect_ratio )
    #     widths = heights / tf.math.square(aspect_ratio)

    #     if tf.random.uniform(()) < 0.5 :
    #         temp = heights
    #         heights = widths
    #         widths = temp
        
    #     else:
    #         temp = heights #for AutoGraph
        
    #     max_width = tf.math.reduce_max(widths)
    #     max_height = tf.math.reduce_max(heights)

    #     x1s = tf.random.uniform((self.batch_size,), minval = 0, maxval = max_width/2 - 0.00001)
    #     y1s = tf.random.uniform((self.batch_size,), minval = 0, maxval = max_height/2 - 0.00001)

    #     x2s = widths + x1s
    #     y2s = heights + y1s

    #     x2s = tf.clip_by_value(x2s, clip_value_min=0, clip_value_max=1.0)
    #     y2s = tf.clip_by_value(y2s, clip_value_min=0, clip_value_max=1.0)

    #     boxes = tf.stack([y1s, x1s, y2s, x2s])

    #     boxes = tf.transpose(boxes)

    #     return boxes

    # @tf.function
    # def random_sized_crop(self, 
    #     example: dict,
    #     min_area: float = 0.25) -> dict:
    #     """
    #     Takes a random crop of image having a random aspect ratio. Resizes it 
    #     to self.image_size. Aspect ratio is NOT maintained. 
    #     Args:
    #         example: A dataset example dict.
    #         min_area: Minimum area of image to be used
    #     Returns:
    #         Example of same format as _TFRECS_FORMAT
    #     """

    #     image = example['image']
    #     h = example['height']
    #     w = example['width']

    #     aspect_ratio = tf.random.uniform((), minval = 3./4., maxval = 4./3.)
    #     area = tf.random.uniform((), minval = min_area, maxval = 1)

    #     boxes = self._get_boxes(aspect_ratio, area)

    #     image = tf.image.crop_and_resize(
    #         image,
    #         boxes,
    #         tf.range(self.batch_size),
    #         (self.image_size[0], self.image_size[0]),
    #     )

    #     return {
    #         "image": tf.cast(image,tf.uint8),
    #         "height": self.image_size,
    #         "width": self.image_size,
    #         "filename": example["filename"],
    #         "label": example["label"],
    #         "synset": example["synset"],
    #     }

    
    def _one_hot_encode_example(self, example: dict) -> dict:
        """Takes an example having keys 'image' and 'label' and returns example
        with keys 'image' and 'target'. 'target' is one hot encoded.
        Args:
            example: an example having keys 'image' and 'label'
        Returns:
            example having keys 'image' and 'target'
        """
        return (example["image"], tf.one_hot(example["label"], self.num_classes))

    
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

        #batch shape: (128, 512, 512, 3)

        # if self.augment_fn == "default":
        #     pass

        # else:
        #     ds = ds.map(
        #         lambda example: self.augment_fn(example),
        #         num_parallel_calls = tf.data.AUTOTUNE
        #     )

        if self.randaugment:
            ds = ds.map(
                lambda example: self._randaugment(example),
                num_parallel_calls = tf.data.AUTOTUNE
            )
        
        ds = ds.map(
            lambda example: self._one_hot_encode_example(example),
            num_parallel_calls = tf.data.AUTOTUNE
        )


        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds