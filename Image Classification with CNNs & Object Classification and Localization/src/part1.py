from .train import train
from .test import test
from .dataset import create_dataset
from .models import create_custom_model, get_VGG
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
def part1(args, model_name):
    match model_name:
        case 'custom':
            create_model = create_custom_model
        case 'vgg':
            create_model = get_VGG
        
    # arguments
    data_dir = args.data_dir
    train_paths_file = args.train_paths_file
    test_paths_file = args.test_paths_file
    batch_sizes = args.batch_size
    learning_rates = args.learning_rate
    verbose = args.verbose
    p = 0.
    metrics = [CategoricalAccuracy(), Precision(), Recall()]

    for batch_size in batch_sizes:
        train_dataset = create_dataset(txt_path=train_paths_file,
                                batch_size=batch_size,
                                data_dir=data_dir)
    
        test_dataset = create_dataset(txt_path=test_paths_file,
                                    batch_size=batch_size,
                                    data_dir=data_dir)
        for learning_rate in learning_rates:
            if verbose:
                print(f'BATCH SIZE:\t{batch_size}')
                print(f'LEARNING RATE:\t{learning_rate}')
                print('A MODEL WILL BE TRAINED')
                print('-------------------------')
            
            sample_batch = next(iter(train_dataset))
            logit_size = sample_batch[1].shape[1]
            if verbose:
                print('DATASETS ARE CREATED SUCCESSFULLY')
                print(logit_size)

            model = create_model(activation_function='relu', 
                                logit_size=logit_size)
            
            if verbose:
                print('CUSTOM MODEL CREATED SUCCESSFULLY')

            learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=10000,
                decay_rate=0.96
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

            checkpoint = f'src/checkpoint/part1_{model_name}_{p}_{learning_rate}_{batch_size}_best_model.h5'
            if os.path.exists(checkpoint):
                print('MODEL EXISTS')
                print('TESTING MODEL')
                model = load_model(checkpoint)
                test(
                    model=model,
                    dataset=test_dataset,
                    model_name=model_name,
                    p=p,
                    lr=learning_rate,
                    batch_size=batch_size,
                    test_file=f'src/results/part1_{model_name}_{p}_{learning_rate}_{batch_size}.txt'
                )
                continue
            
            log_dir = f'src/logs/part1_{model_name}_{p}_{learning_rate}_{batch_size}'
            hist_file = f'src/histories/part1_{model_name}_{p}_{learning_rate}_{batch_size}.json'
            
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

            if verbose:
                print('MODEL IS TRAINED')
                print('MODEL IS BEING TESTED.')
            
            model = load_model(checkpoint)
            
            test(
                model=model,
                dataset=test_dataset,
                model_name=model_name,
                p=p,
                lr=learning_rate,
                batch_size=batch_size,
                test_file=f'src/results/part1_{model_name}_{p}_{learning_rate}_{batch_size}.txt'
            )

            if verbose:
                print('MODEL IS TESTED.')
                print('=================================')

def optimal_hyper_parameters(model_name):
    paths = os.listdir('src/results')

    optimal_learning_rate = -1
    optimal_batch_size = -1
    optimal_dropout_rate = -1
    max_value = 0
    for path in paths:
        _, name, dropout_rate, learning_rate, batch_size = path[:-4].split('_')
        if name != model_name:
           continue
        learning_rate = float(learning_rate)
        batch_size = int(batch_size)

        path = os.path.join('src/results', path)

        with open(path, 'r') as f:
            loss = float(f.readline().split(':')[-1])
            accuracy = float(f.readline().split(':')[-1])
            precision = float(f.readline().split(':')[-1])
            recall = float(f.readline().split(':')[-1])
        t = precision * recall
        if max_value < t:
            max_value= t
            optimal_learning_rate = learning_rate
            optimal_batch_size = batch_size
            optimal_dropout_rate = dropout_rate
    return optimal_learning_rate, optimal_batch_size, float(optimal_dropout_rate)


def check_dropout(args, model_name='custom'):
    print(optimal_hyper_parameters('custom'))
    print(optimal_hyper_parameters('vgg'))

    # arguments
    data_dir = args.data_dir
    train_paths_file = args.train_paths_file
    test_paths_file = args.test_paths_file
    verbose = args.verbose
    dropout_rates = args.dropout_rate

    learning_rate, batch_size, _ = optimal_hyper_parameters('custom')
    train_dataset = create_dataset(txt_path=train_paths_file,
                                batch_size=batch_size,
                                data_dir=data_dir)
    
    test_dataset = create_dataset(txt_path=test_paths_file,
                                batch_size=batch_size,
                                data_dir=data_dir)
    
    sample_batch = next(iter(train_dataset))
    logit_size = sample_batch[1].shape[1]

    for p in dropout_rates:
        model = create_custom_model(logit_size=logit_size, p=p)

        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            decay_rate=0.96
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
        metrics = [CategoricalAccuracy(), Precision(), Recall()]

        checkpoint = f'src/checkpoint/part1_{model_name}_{p}_{learning_rate}_{batch_size}_best_model.h5'
        if os.path.exists(checkpoint):
            print('MODEL EXISTS')
            print('TESTING MODEL')
            model = load_model(checkpoint)
            test(
                model=model,
                dataset=test_dataset,
                model_name=model_name,
                p=p,
                lr=learning_rate,
                batch_size=batch_size,
                test_file=f'src/results/part1_{model_name}_{p}_{learning_rate}_{batch_size}.txt'
            )
            continue
        
        log_dir = f'src/logs/part1_{model_name}_{p}_{learning_rate}_{batch_size}'
        hist_file = f'src/histories/part1_{model_name}_{p}_{learning_rate}_{batch_size}.json'
        
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
            model_name=model_name,
            p=p,
            lr=learning_rate,
            batch_size=batch_size,
            test_file=f'src/results/part1_{model_name}_{p}_{learning_rate}_{batch_size}.txt'
        )