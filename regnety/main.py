import tensorflow as tf
import os
import random
import json
import argparse
import math
from utils.tfrecords_utils import *
from dataset.imagenet import ImageNet
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Make TFRecords")
    parser.add_argument("--odir", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--file_prefix", type=str)
    parser.add_argument("--synset_filepath", type=str)
    parser.add_argument("--batch_size", type=int, default=1024)

    args = parser.parse_args()
    make_tfrecs(
        dataset_base_dir=args.data_dir,
        output_dir=args.odir,
        file_prefix=args.file_prefix,
        synset_filepath=args.synset_filepath,
        batch_size = args.batch_size,
    )
    # for i in ds:
    #     print(i)
    #     break
        

    imgnet = ImageNet([os.path.join('/content',i) for i in os.listdir('/content') if i.startswith('trial5')])
    ds = imgnet.make_dataset()
    for i in ds:
      print(i)
      im = i['image']/255.
      plt.imshow(im)
      plt.savefig('image.jpeg')
      break


main()
