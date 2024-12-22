from .train import train
from .test import test
from .dataset import create_dataset
from .models import get_VGG
from .part1 import optimal_hyper_parameters
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def part2(args):
    data_dir = args.data_dir
    train_paths_file = args.train_paths_file
    test_paths_file = args.test_paths_file
    verbose = args.verbose
    
    learning_rate, batch_size, dropout_rate = optimal_hyper_parameters('vgg')

    train_dataset = create_dataset(txt_path=train_paths_file,
                                batch_size=batch_size,
                                data_dir=data_dir)
    
    test_dataset = create_dataset(txt_path=test_paths_file,
                                batch_size=batch_size,
                                data_dir=data_dir)
    
    sample_batch = next(iter(train_dataset))
    logit_size = sample_batch[1].shape[1]
    dropout_rate = float(dropout_rate)

    model = get_VGG(logit_size,
                    p=dropout_rate,
                    unfreeze_convs=True)
    
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            decay_rate=0.96
        )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    metrics = [CategoricalAccuracy(), Precision(), Recall()]

    checkpoint = f'src/checkpoint/part2_vgg_{dropout_rate}_{learning_rate}_{batch_size}_best_model.h5'
    log_dir = f'src/logs/part2_vgg_{dropout_rate}_{learning_rate}_{batch_size}'
    hist_file = f'src/histories/part2_vgg_{dropout_rate}_{learning_rate}_{batch_size}.json'

    train(
        model=model,
        dataset=train_dataset,
        optimizer=optimizer,
        metrics=metrics,
        epochs=100,
        checkpoint=checkpoint,
        log_dir=log_dir,
        hist_file=hist_file,
    )
    
    test(
        model=model,
        dataset=test_dataset,
        test_file=f'src/results/part2_vgg_{dropout_rate}_{learning_rate}_{batch_size}.txt'
    )
    


