import apache_beam as beam
import tensorflow as tf

from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from regnety.regnety.utils.image_utils import *

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = (
            value.numpy()
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_collection(list1, list2, list3):

    final_list = [
        (list1[i], list2[i], list3[i]) for i in range(len(list1))
    ]


    with beam.Pipeline() as pipeline:
        coll = (
            pipeline
            | beam.Create(final_list))

    return coll


class MakeImageDoFn(beam.DoFn):
    def process(
        self, 
        info_tuple
    ):
        filepath, label, synset = info_tuple
        image_str = tf.io.read_file(filepath)

        if is_png(filepath):
            image_str = png_to_jpeg(image_str)

        if is_cmyk(filepath):
            image_str = cmyk_to_rgb(image_str)

        image_tensor = tf.io.decode_jpeg(image_str)
        height, width = image_tensor.shape[0], image_tensor.shape[1]

        if not is_rgb(image_tensor):
            image_tensor = tf.image.grayscale_to_rgb(image_tensor)
    
        image_tensor = tf.cast(tf.image.resize(image_tensor, (512,512)), tf.uint8)

        image_str = tf.io.encode_jpeg(image_tensor)

        assert len(image_tensor.shape) == 3

        try:
            return  tf.train.Example(
                features=tf.train.Features(
                    feature = {
                        "image": _bytes_feature(image_str),
                        "height": _int64_feature(height),
                        "width": _int64_feature(width),
                        "filename": _bytes_feature(
                            bytes(os.path.basename(filepath)).encode("utf8")
                        ),
                        "label": _int64_feature(label),
                        "synset": _bytes_feature(bytes(synset).encode("utf8")),
                    }
                )
            )
        except:
            return tf.train.Example(
                features=tf.train.Features(
                    feature = {
                        "image": _bytes_feature(image_str),
                        "height": _int64_feature(height),
                        "width": _int64_feature(width),
                        "filename": _bytes_feature(
                            bytes(os.path.basename(filepath), encoding="utf8")
                        ),
                        "label": _int64_feature(label),
                        "synset": _bytes_feature(bytes(synset, encoding="utf8")),
                    }
                )
            )
    
    def __call__(self, *args):
        self.process(*args)

class MakeExampleDoFn(beam.DoFn):
    def process(self,
        example
    ):
        
        return example.SerializeToString()
    
    def __call__(self, *args):
        self.process(*args)

