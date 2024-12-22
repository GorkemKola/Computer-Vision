import tensorflow as tf
from .dataset import create_dataset
from .models import create_custom_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import json

def train(
        model, 
        dataset, 
        epochs=100,
        optimizer='adam', 
        loss_fn = 'categorical_crossentropy',
        metrics=['accuracy'],
        checkpoint = None,
        log_dir=None,
        hist_file='training_history.json',
    ):

    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=metrics)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=30,
        restore_best_weights=True
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint, 
        save_best_only=True
    )

    tensorboard_callback = TensorBoard(
        log_dir=log_dir
    )

    all_callbacks = [early_stopping, checkpoint_callback, tensorboard_callback]

    split = int(0.8 * len(dataset)) 
    train_dataset = dataset.take(split)
    val_dataset = dataset.skip(split)

    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # Train the model
        history = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=val_dataset,
                    validation_steps=len(val_dataset),
                    callbacks=all_callbacks)
    with open(hist_file, 'w') as f:
        json.dump(history.history, f)
    return history

if __name__ == '__main__':
    train_dataset = create_dataset('src/data/TrainImages.txt', 16, data_dir='src/data/indoorCVPR_09/Images')
    model = create_custom_model('relu', 67)
    
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.96
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    metrics = [tf.keras.metrics.CategoricalAccuracy(), 
               tf.keras.metrics.Precision(), 
               tf.keras.metrics.Recall()]

    train(
        model=model,
        dataset=train_dataset,
        optimizer=optimizer,
        metrics=metrics,
        epochs=100,
        checkpoint='src/checkpoint/part1_best_model.h5',
        log_dir='src/logs/part1',
        hist_file='src/histories/part1'
    )
