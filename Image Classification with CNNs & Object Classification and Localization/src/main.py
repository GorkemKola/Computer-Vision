import argparse
import yaml
import tensorflow as tf
from .part1 import part1, check_dropout
from .part2 import part2
from .part3 import part3
from .dataset import clean_data, create_paths, write_paths 

def main(args):
    if args.clean_data:
        labels = clean_data(args.data_dir)
        train_paths, test_paths = create_paths(args.data_dir, labels)
        write_paths(train_paths, test_paths, args.train_paths_file, args.test_paths_file)
    if args.verbose:
        print('TRAIN AND TEST IMAGES SELECTED.')

    if args.part1:
        part1(args, 'custom')
        part1(args, 'vgg')
        check_dropout(args)
    if args.part2:
        part2(args)
    if args.part3:
        part3(args)
if __name__ == "__main__":

    with open('src/config.yml') as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument("--data_dir", 
                        default=config['DATADIR'], 
                        help="Path to the Image Dataset")
    
    parser.add_argument("--train_paths_file", 
                        default=config['TRAIN_PATHS'], 
                        help="Path to txt file that contains train images paths.")
    
    parser.add_argument("--test_paths_file", 
                    default=config['TEST_PATHS'], 
                    help="Path to txt file that contains test images paths.")

    parser.add_argument('--part1', 
                        action="store_true", 
                        default=False, 
                        help="Run Part 1 Training")
    
    parser.add_argument('--part2', 
                        action="store_true", 
                        default=False, 
                        help="Run Part 2 Training")
    
    parser.add_argument('--part3', 
                        action="store_true", 
                        default=False, 
                        help="Run Part 3 Training")

    
    parser.add_argument('--clean_data', 
                        action="store_true", 
                        default=False, 
                        help="Selects 15 class that has more than 250 instances.")

    parser.add_argument('--batch_size',
                        default=config['BATCH_SIZE'],
                        nargs='+',
                        type=int,
                        help='Adjust the batch sizes as list')
    
    parser.add_argument('--learning_rate',
                        default=config['LEARNING_RATE'],
                        nargs='+',
                        type=float,
                        help='Adjust the learning rates as list')

    parser.add_argument('--dropout_rate',
                        default=config['LEARNING_RATE'],
                        nargs='+',
                        type=float,
                        help='Adjust the dropoutrates as list')

    parser.add_argument("--verbose", 
                        action="store_true", 
                        help="Enable verbose mode")

    args = parser.parse_args()

    if args.verbose:
        print('HyperParameters ARE DEFINED AS:')
        print('====================')
        print(f'BATCH SIZES:\t{args.batch_size}')
        print(f'LEARNING RATES:\t{args.learning_rate}')
    main(args)
