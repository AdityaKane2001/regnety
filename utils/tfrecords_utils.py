"""Script to make and save TFRecords from ImageNet files"""

import tensorflow as tf
import os


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# make_tfrecs -> for(make_single_tfrecord)-> make_dataset -> make_example 
def get_synset_labels(filepath):
  with open(filepath,'r') as f:
    all_lines = f.readlines()
    all_lines.sort()
    all_lines = list(map(lambda x: x.split(': '), all_lines))

    labels_dict = dict()

    for i in range(len(all_lines)):
      labels_dict[all_lines[i][0]] = (i,all_lines[i][1])
  
  return labels_dict



def get_files(data_dir):
  all_images = tf.io.gfile.glob(os.path.join(data_dir,'*','*.JPEG'))
  all_synsets = [os.path.basename(os.path.dirname(f)) for f in all_images]

  all_indexes = random.shuffle([range(len(all_images))])
  
  all_images = [all_images[i] for i in all_indexes]
  all_synsets = [all_synsets[i] for i in all_indexes]

  labels_dict = get_synset_labels('/content/regnety/config/imagenet_synset_to_human.txt')

  all_labels_int = [labels_dict[i][0] for i in all_synsets]

  return all_images, all_labels_int, all_synsets




def make_tfrecs(dataset_base_dir = None , #example: home/imagenet/train
                output_dir = None, #example: home/imagenet_tfrecs 
                file_prefix = None, #example: file_prefix = train makes all files look like: train_0000_of_num_shards 
                num_shards = 1024):

  """Driver function"""

  images, labels, synsets = get_files('/content/imagenette2/train')

  chunksize = int(math.ceil(len(images) / num_shards))




