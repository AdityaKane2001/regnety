import tensorflow as tf
import os
import random
import json
import argparse
import math
from utils.tfrecords_utils import *
from dataset.imagenet import ImageNet

def main():
  parser = argparse.ArgumentParser(description='Make TFRecords')
  parser.add_argument('--odir', type=str)
  parser.add_argument('--data_dir',type=str)
  parser.add_argument('--file_prefix',type=str)
  parser.add_argument('--synset_filepath',type=str)
  parser.add_argument('--shards',type=int,default=10)

  args = parser.parse_args()
  # make_tfrecs(dataset_base_dir=args.data_dir,
  #             output_dir =  args.odir,
  #             file_prefix =  args.file_prefix,
  #             synset_filepath = args.synset_filepath,
  #             num_shards = args.shards)
  imgnet = ImageNet([os.path.join('/content',i) for i in os.listdir('/content') if i.startswith('trial4')])
  ds = imgnet.make_dataset()
  for i in ds:
    print(i)
    break
main()