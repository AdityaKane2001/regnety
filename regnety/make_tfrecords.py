import argparse
from utils.tfrecords_utils import *


def main():
    parser = argparse.ArgumentParser(description="Make TFRecords")
    parser.add_argument("-o", "--odir", type=str,)
    parser.add_argument("-d", "--data_dir", type=str)
    parser.add_argument("-fpr", "--file_prefix", type=str)
    parser.add_argument("-synpath","--synset_filepath", type=str)
    parser.add_argument("-bs", "--batch_size", type=int, default=1024)
    parser.add_argument("-f", "--log_freq",type=int, default=50)
    parser.add_argument("-s", "--shuffle", action='store_true')

    args = parser.parse_args()
    make_tfrecs(
        dataset_base_dir=args.data_dir,
        output_dir=args.odir,
        file_prefix=args.file_prefix,
        synset_filepath=args.synset_filepath,
        batch_size = args.batch_size,
        logging_frequency = args.log_freq,
        shuffle = args.shuffle
    )

if __name__=='__main__':
    main()