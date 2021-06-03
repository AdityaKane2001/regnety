"""Script to make and save TFRecords from ImageNet files"""

import tensorflow as tf
import os
import random
import json
import argparse
import math
from image_utils import *

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
    raw_labels_dict = json.load(f)
  labels_dict = dict()

  for i in raw_labels_dict:
    labels_dict[raw_labels_dict[i]['id']] = (int(i), raw_labels_dict[i]['label'])
  # with open(filepath,'r') as f:
  #   all_lines = f.readlines()
  #   all_lines.sort()
  #   all_lines = list(map(lambda x: x.split(': '), all_lines))

  #   labels_dict = dict()

  #   for i in range(len(all_lines)):
  #     labels_dict[all_lines[i][0]] = (i,all_lines[i][1])
  
  return labels_dict



def get_files(data_dir):
  all_images = tf.io.gfile.glob(os.path.join(data_dir,'*','*.JPEG'))
  all_synsets = [os.path.basename(os.path.dirname(f)) for f in all_images]

  print(len(all_images))
  all_indexes = list(range(len(all_images))) 
  random.shuffle(all_indexes)
  
  all_images = [all_images[i] for i in all_indexes]
  all_synsets = [all_synsets[i][1:]+'-n' for i in all_indexes]

  labels_dict = get_synset_labels('/content/regnety/config/imagenet_synset_to_human.json')

  all_labels_int = [labels_dict[i][0] for i in all_synsets]

  return all_images, all_labels_int, all_synsets


def make_image(filepath):

  with tf.io.gfile.GFile(filepath,'rb') as f:
    image_str = f.read()
  
  if is_png(filepath):
    image_str = png_to_jpeg(image_str)
  
  if is_cmyk(filepath):
    image_str = cmyk_to_rgb(image_str)
  
  image_tensor = tf.io.decode_jpeg(image_str)

  height = image_tensor.shape[0]
  width = image_tensor.shape[1]

  if image_tensor.shape[2] == 1:
    image_tensor = tf.image.grayscale_to_rgb(image_tensor)
    image_str = tf.io.encode_jpeg(image_tensor)

  
  assert len(image_tensor.shape) == 3
  assert image_tensor.shape[2] == 3

  return image_str, height, width


def make_example(image_str,height,width,filepath,label,synset):

  example = tf.train.Example(features = tf.train.Features(feature={
    'image' : _bytes_feature(image_str),
    'height' : _int64_feature(height),
    'width' : _int64_feature(width),
    'filename' : _bytes_feature(bytes(os.path.basename(os.path.dirname(filepath)),encoding='utf8')),
    'label' : _int64_feature(label),
    'synset' : _bytes_feature(bytes(synset,encoding='utf8'))
  }))

  return example


def make_single_tfrecord(chunk_files,chunk_synsets,chunk_labels,output_filepath):
  
  with tf.io.TFRecordWriter(output_filepath) as writer:
    for i in range(len(chunk_files)):
      image_str, height, width = make_image(chunk_files[i])
      label = chunk_labels[i]
      synset = chunk_synsets[i]

      example = make_example(image_str,height,width,chunk_files[i],label,synset)

      writer.write(example.SerializeToString())
  writer.close()




def make_tfrecs(dataset_base_dir = None , #example: home/imagenet/train
                output_dir = None, #example: home/imagenet_tfrecs 
                file_prefix = None, #example: file_prefix = 'train' makes all files look like: train_0000_of_num_shards 
                num_shards = 10):

  """Driver function"""

  images, labels, synsets = get_files(dataset_base_dir)

  chunksize = int(math.ceil(len(images) / num_shards))

  for shard in range(num_shards):
    chunk_files = images[shard * chunksize : (shard + 1) * chunksize]
    chunk_synsets = synsets[shard * chunksize : (shard + 1) * chunksize]
    chunk_labels = labels[shard * chunksize : (shard + 1) * chunksize]

    output_filepath = os.path.join(output_dir,file_prefix+'_%.4d_of_%.4d'%(shard,num_shards))

    make_single_tfrecord(chunk_files,chunk_synsets,chunk_labels,output_filepath)
    
def main():
  parser = argparse.ArgumentParser(description='Make TFRecords')
  parser.add_argument('--odir', type=str)
  parser.add_argument('--data_dir',type=str)
  parser.add_argument('--file_prefix',type=str)
  parser.add_argument('--shards',type=int,default=10)

  args = parser.parse_args()
  make_tfrecs(dataset_base_dir=args.data_dir,
              output_dir =  args.odir,
              file_prefix =  args.file_prefix,
              num_shards = args.shards)

main()

