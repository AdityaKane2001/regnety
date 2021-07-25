import argparse
from utils.tfrecords_utils import *


def main():
    parser = argparse.ArgumentParser(description="Make TFRecords")
    parser.add_argument("-o", "--odir", type=str,
        help="Output directory for TFRecords.")
    parser.add_argument("-d", "--data_dir", type=str,
        help="Input data directory. Must have ImageNet-like directory structure as shown in https://gist.github.com/AdityaKane2001/490b8c94c05538dec690513022195e91")
    parser.add_argument("-fpr", "--file_prefix", type=str,
        help="File prefix to add to all files.")
    parser.add_argument("-synpath","--synset_filepath", type=str, default="",
        help="Path to JSON file containing synsets. Default file is used if unspecified.")
    parser.add_argument("-bs", "--batch_size", type=int, default=1024,
        help="Batch size for the dataset. One shard contains these many examples.")
    parser.add_argument("-f", "--log_freq",type=int, default=50,
        help="`Writing shard..` will be printed after these many shards.") 
    parser.add_argument("-s", "--shuffle", action="store_true",
        help="Shuffle dataset before making TFRecords.")
    parser.add_argument("-v", "--validation_set",action="store_true",
        help="To be specified if dataset has the file structure of ImageNet validation set.")
    parser.add_argument("-b", "--use_apache_beam",action="store_true",
        help="To be specified if Apache Beam is to be used to make TFRecords.")
    

    args = parser.parse_args()
    if args.use_apache_beam:
        make_tfrecs_beam(
            dataset_base_dir=args.data_dir,
            output_dir=args.odir,
            file_prefix=args.file_prefix,
            synset_filepath=args.synset_filepath,
            batch_size = args.batch_size,
            logging_frequency = args.log_freq,
            shuffle = args.shuffle,
            val = args.validation_set
        )
    else:
        make_tfrecs(
            dataset_base_dir=args.data_dir,
            output_dir=args.odir,
            file_prefix=args.file_prefix,
            synset_filepath=args.synset_filepath,
            batch_size = args.batch_size,
            logging_frequency = args.log_freq,
            shuffle = args.shuffle,
            val = args.validation_set
        )

if __name__=="__main__":
    main()
