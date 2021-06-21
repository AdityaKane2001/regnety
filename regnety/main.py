import argparse
import matplotlib.pyplot as plt
import os
import shutil
import time

from dataset.imagenet import ImageNet
from datetime import timedelta
from utils.tfrecords_utils import *

def main():
    parser = argparse.ArgumentParser(description="Make TFRecords")
    parser.add_argument("-o", "--odir", type=str,)
    parser.add_argument("-d", "--data_dir", type=str)
    parser.add_argument("-fpr", "--file_prefix", type=str)
    parser.add_argument("-synpath","--synset_filepath", type=str, default='')
    parser.add_argument("-bs", "--batch_size", type=int, default=1024)
    parser.add_argument("-f", "--log_freq",type=int, default=50)
    parser.add_argument("-g", "--log_gap",type=int, default=3600)
    parser.add_argument("-s", "--shuffle", action='store_true')
    

    args = parser.parse_args()
    start_time = time.time()
    make_tfrecs(
        dataset_base_dir=args.data_dir,
        output_dir=args.odir,
        file_prefix=args.file_prefix,
        synset_filepath=args.synset_filepath,
        batch_size = args.batch_size,
        logging_frequency = args.log_freq,
        logging_gap = args.log_gap,
        shuffle = args.shuffle
    )
    end_time = time.time()
    print()
    td = timedelta(seconds = end_time - start_time)
    h = td.seconds // 3600
    m = (td.seconds/60) % 60
    s = td.seconds % 60
    print('Time taken for 9469 images is %d hours, %d minutes and %d seconds.'
        % (h,m,s))

    td = timedelta(seconds = end_time - start_time)
    h = td.seconds // 3600
    m = (td.seconds/60) % 60
    s = td.seconds % 60

    td = td * (1330000/9469.)
    h = td.seconds // 3600
    m = (td.seconds/60) % 60
    s = td.seconds % 60
    print('Thus, time taken for ImageNet1k will be %d hours, %d minutes and %d seconds'
        % (h, m, s))
    print()

    imgnet = ImageNet(
        [
            os.path.join("/content", i)
            for i in os.listdir("/content")
            if i.startswith("trial")
        ]
    )
    ds = imgnet.make_dataset()
    for i in ds.take(10):
        filename = i['filename'].numpy().decode('utf8')
        synset = i['synset'].numpy().decode('utf8')
        synset = 'n'+synset[:-2]
        
        shutil.copy(os.path.join(args.data_dir, synset, filename),
            os.path.join('/content','original_'+filename))

        im = i["image"] / 255.0
        plt.imshow(im)
        plt.savefig(os.path.join('/content','augmented_'+filename))


main()
