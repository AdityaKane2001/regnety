import tensorflow as tf
import os


class ImageNet:

  TFRECS_FORMAT = {
    'image' : tf.io.FixedLenFeature([],tf.string),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'filename' : tf.io.FixedLenFeature([],tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'synset': tf.io.FixedLenFeature([], tf.string)
  }


  def __init__(self,tfrecs_filepath = None,
               batch_size = 128,
               image_size = 224,
               augment_fn = 'default'):

    if tfrecs_filepath is None:
      raise ValueError('List of TFrecords paths cannot be None')
    self.tfrecs_filepath = tfrecs_filepath
    self.batch_size = batch_size
    self.image_size = image_size
    self.augment_fn = augment_fn

  
  def _read_tfrecs(self):
    def decode_example(example):
      image = tf.cast(tf.io.decode_jpeg(example['image']),tf.float32)
      height = example['height']
      width = example['width']
      filename = example['filename']
      label = example['label']
      return {
        'image' : image,
        'height' : height,
        'width' : width,
        'filename' :filename,
        'label' : label
      }
      
    ds = tf.data.TFRecordDataset(self.tfrecs_filepath)
    ds = ds.map(lambda example: tf.io.parse_example(example, ImageNet.TFRECS_FORMAT))
    ds = ds.map(lambda example: decode_example(example))
    return ds


  def _scale_and_center_crop(self, image, h, w, scale_size, final_size):
    if w < h and w != scale_size:
        w, h = tf.cast(scale_size, tf.int64), tf.cast(( (h / w) * scale_size), tf.int64)
        im = tf.image.resize(image, (w,h))
    elif h <= w and h != scale_size:
        w, h = tf.cast(( (h / w) * scale_size), tf.int64), tf.cast(scale_size, tf.int64)
        im = tf.image.resize(image, (w,h))
    else:
      im = tf.image.resize(image, (w,h))
    x = tf.cast(tf.math.ceil((w - final_size) / 2 ), tf.int64)
    y = tf.cast(tf.math.ceil((h - final_size) / 2), tf.int64)
    return im[y : (y + final_size), x : (x + final_size), :]


  def _random_sized_crop(self, example, min_area = 0.08, max_iter = 10):
    h, w = tf.cast(example['height'], tf.int64), tf.cast(example['width'], tf.int64)
    area = h * w
    return_example = False
    image = example['image']
    num_iter = tf.constant(0, dtype = tf.int32)
    while num_iter <= max_iter:
      num_iter  = tf.math.add(num_iter,1)
      final_area = tf.random.uniform((), minval = min_area, maxval = 1,
                                      dtype = tf.float32) * tf.cast(area, tf.float32)
      aspect_ratio = tf.random.uniform((), minval = 3./4., maxval = 4./3.,
                                      dtype = tf.float32)
      w_cropped = tf.cast(tf.math.round(tf.math.sqrt(final_area * aspect_ratio)), tf.int64)
      h_cropped = tf.cast(tf.math.round(tf.math.sqrt(final_area / aspect_ratio)), tf.int64)

      if tf.random.uniform(()) < 0.5:
        w_cropped, h_cropped = h_cropped, w_cropped
      
      if h_cropped <= h and w_cropped <= w:
        return_example = True
        if h_cropped == h: 
          y = tf.constant(0, dtype = tf.int64)
        else:
          y = tf.random.uniform((),minval = 0, maxval = h - h_cropped, dtype = tf.int64)
        if w_cropped == w:
          x = tf.constant(0, dtype = tf.int64)
        else:
          x = tf.random.uniform((),minval = 0, maxval = w - w_cropped, dtype = tf.int64)
      
        image = image[y : (y + h_cropped), x : x + w_cropped, :]
        image = tf.image.resize(image, (self.image_size, self.image_size))
        break
    if return_example:
      return {
          'image' : image,
          'height' : self.image_size,
          'width' : self.image_size,
          'filename' : example['filename'],
          'label' : example['label']
        }

    image = self._scale_and_center_crop(image, h, w, self.image_size, self.image_size)
    return {
          'image' : image,
          'height' : self.image_size,
          'width' : self.image_size,
          'filename' : example['filename'],
          'label' : example['label']
        }
    
 
  def make_dataset(self, ):
    ds = self._read_tfrecs()

    if self.augment_fn == 'default':
      ds = ds.map(lambda example: self._random_sized_crop(example))

    else:
      ds = ds.map(lambda example: self.augment_fn(example))
    return ds
