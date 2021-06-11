import tensorflow as tf
import os
from official.vision.image_classification.augment import RandAugment


class ImageNet:
    """Class for all ImageNet related functions, includes TFRecords loading,
    preprocessing and augmentations. TFRecords must follow the format given
    below.

    Args:
        tfrecs_filepath: list of filepaths of all TFRecords files
        batch_size: batch_size for the Dataset
        image_size: final image size of the images in the dataset
        augment_fn: function to apply to dataset after loading raw TFrecords
                    'default' implies following augmentations will be applied:
                    - Random sized crop (train only)
                    - Scale and center crop
                    - RandAugment (TODO)
        num_classes: number of classes
        randaugment: True of RandAugment is to be applied

    Class attributes:
        TFRECS_FORMAT: Expected format of TFRecords stored.
        _MEAN: Channel wise mean for ImageNet
        _STD: Channel wise standard deviation for ImageNet

    """

    TFRECS_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "filename": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "synset": tf.io.FixedLenFeature([], tf.string),
    }

    _MEAN = tf.constant([0.485, 0.456, 0.406])
    _STD = tf.constant([0.229, 0.224, 0.225])

    def __init__(
        self,
        tfrecs_filepath=None,
        batch_size=128,
        image_size=224,
        augment_fn="default",
        num_classes=10,
        randaugment=True,
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
            self.augmenter = RandAugment(magnitude=5)

    def decode_example(self, example):
        """Decodes an example to its individual attributes

        Args:
            example: A TFRecord dataset example.

        Returns:
            Dict containing attributes from a single example. Follows
            the same names as TFRECORDS_FORMAT.
        """
        image = tf.cast(tf.io.decode_jpeg(example["image"]), tf.float32)
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

    def _read_tfrecs(self):
        """Function for reading and loading TFRecords into a tf.data.Dataset.

        Returns:
            A tf.data.Dataset
        """
        ds = tf.data.TFRecordDataset(self.tfrecs_filepath)
        ds = ds.map(
            lambda example: tf.io.parse_example(example, ImageNet.TFRECS_FORMAT)
        )
        ds = ds.map(lambda example: self.decode_example(example))
        return ds

    def _scale_and_center_crop(self, image, h, w, scale_size, final_size):
        """Resizes image to given scale size and returns a center crop. Aspect
        ratio is NOT maintained.

        Args:
            image: tensor of the image
            h: initial height of image
            w: initial width of image
            scale_size: Size of image to scale to
            final_size: Size of final image

        Returns:
            Tensor of shape (final_size, final_size, 3)
        """
        if w < h and w != scale_size:
            w, h = tf.cast(scale_size, tf.int64), tf.cast(
                ((h / w) * scale_size), tf.int64
            )
            im = tf.image.resize(image, (w, h))
        elif h <= w and h != scale_size:
            w, h = tf.cast(((h / w) * scale_size), tf.int64), tf.cast(
                scale_size, tf.int64
            )
            im = tf.image.resize(image, (w, h))
        else:
            im = tf.image.resize(image, (w, h))
        x = tf.cast((w - final_size) / 2, tf.int64)
        y = tf.cast((h - final_size) / 2, tf.int64)
        return im[y : (y + final_size), x : (x + final_size), :]

    def random_sized_crop(self, example, min_area=0.08, max_iter=10):
        """
        Randomly chooses an area and aspect ratio and resizes the image to those
        values.

        Args:
            example: A dataset example.
            min_area: Minimum area of image to be used
            max_iter: Maximum iteration to try before implementing scale and
                center crop. Randomly genereted width and height must be lower
                than original width and height before max_iter.

        Returns:
            Example of same format as TFRECS_FORMAT
        """

        h, w = tf.cast(example["height"], tf.int64), tf.cast(
            example["width"], tf.int64
        )
        area = h * w
        return_example = False
        image = example["image"]
        num_iter = tf.constant(0, dtype=tf.int32)
        while num_iter <= max_iter:
            num_iter = tf.math.add(num_iter, 1)
            final_area = tf.random.uniform(
                (), minval=min_area, maxval=1, dtype=tf.float32
            ) * tf.cast(area, tf.float32)
            aspect_ratio = tf.random.uniform(
                (), minval=3.0 / 4.0, maxval=4.0 / 3.0, dtype=tf.float32
            )
            w_cropped = tf.cast(
                tf.math.round(tf.math.sqrt(final_area * aspect_ratio)), tf.int64
            )
            h_cropped = tf.cast(
                tf.math.round(tf.math.sqrt(final_area / aspect_ratio)), tf.int64
            )

            if tf.random.uniform(()) < 0.5:
                w_cropped, h_cropped = h_cropped, w_cropped

            if h_cropped <= h and w_cropped <= w:
                return_example = True
                if h_cropped == h:
                    y = tf.constant(0, dtype=tf.int64)
                else:
                    y = tf.random.uniform(
                        (), minval=0, maxval=h - h_cropped, dtype=tf.int64
                    )
                if w_cropped == w:
                    x = tf.constant(0, dtype=tf.int64)
                else:
                    x = tf.random.uniform(
                        (), minval=0, maxval=w - w_cropped, dtype=tf.int64
                    )

                image = image[y : (y + h_cropped), x : x + w_cropped, :]
                image = tf.image.resize(
                    image, (self.image_size, self.image_size)
                )
                break
        if return_example:
            return {
                "image": image,
                "height": self.image_size,
                "width": self.image_size,
                "filename": example["filename"],
                "label": example["label"],
                "synset": example["synset"],
            }

        image = self._scale_and_center_crop(
            image, h, w, self.image_size, self.image_size
        )
        return {
            "image": image,
            "height": self.image_size,
            "width": self.image_size,
            "filename": example["filename"],
            "label": example["label"],
            "synset": example["synset"],
        }

    def clip_example(self, example):
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
        example['image'] = self.augmenter.distort(example['image'])
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
            if self.randaugment:
                ds = ds.map(lambda example: self._randaugment(example))
            #ds = ds.map(lambda example: self.clip_example(example))

        else:
            ds = ds.map(lambda example: self.augment_fn(example))

        return ds
