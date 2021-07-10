import apache_beam as beam
import tensorflow as tf

from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

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
        filepath,
        labels_int,
        synset
    ):
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

        return image_str, height, width, filepath, label, synset

class MakeExampleDoFn(beam.DoFn):
    def process(self,
        image_str,
        height,
        width, 
        filepath, 
        label, 
        synset
    ):
        try:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
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
        except TypeError:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
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
        return example
