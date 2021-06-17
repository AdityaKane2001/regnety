import argparse
import matplotlib.pyplot as plt
import os
import time

from dataset.imagenet import ImageNet
from datetime import timedelta
from utils.tfrecords_utils import *

def main():
    parser = argparse.ArgumentParser(description="Make TFRecords")
    parser.add_argument("--odir", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--file_prefix", type=str)
    parser.add_argument("--synset_filepath", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=1024)

    args = parser.parse_args()
    start_time= time.time()
    make_tfrecs(
        dataset_base_dir=args.data_dir,
        output_dir=args.odir,
        file_prefix=args.file_prefix,
        synset_filepath=args.synset_filepath,
        batch_size = args.batch_size,
    )
    end_time = time.time()
    print()
    print('Time taken for 9469 images: ',str(timedelta(seconds = end_time - start_time)))


    imgnet_time = (end_time - start_time) * (1330000/9469.)

    print('Thus, time taken for ImageNet1k will be ' + str(timedelta(seconds = imgnet_time)))
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
