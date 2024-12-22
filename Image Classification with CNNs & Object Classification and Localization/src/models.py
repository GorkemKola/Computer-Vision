import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
def create_custom_model(
        logit_size,
        input_shape=(128, 128, 3),
        activation_function='relu',
        p=.0):
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (7, 7), activation=activation_function, input_shape=input_shape, strides=(2, 2)))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(64, (3, 3), activation=activation_function))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation=activation_function))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())
    
    model.add(layers.Dense(128, activation=activation_function))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(p))
    
    model.add(layers.Dense(logit_size, activation='softmax'))
    
    return model

def get_VGG(
        logit_size,
        input_shape=(128, 128, 3),
        activation_function='relu',
        unfreeze_convs=False,
        p = .0
            ):
    base_model = VGG16(weights='imagenet', 
                       include_top=False, 
                       input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
        
    if unfreeze_convs:
        for layer in list(reversed(base_model.layers))[:3]:
            layer.trainable = True

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation=activation_function))
    model.add(layers.Dropout(p))
    model.add(layers.Dense(logit_size, activation='softmax'))

    return model


def build_detector(
        logit_size,
        activation_function='relu',
        p=0,
        ):
    
    inputs = layers.Input(shape=(224, 224, 3))
    # Load pre-trained ResNet50 model
    backbone = ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_tensor=inputs
        )
    
    x = backbone.output
    x = layers.Conv2D(256, kernel_size=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation_function)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(p)(x)
    bbox = layers.Dense(4, activation="sigmoid", name="bbox")(x)
    label = layers.Dense(logit_size, activation="softmax", name="label")(x)

    model = Model(inputs=[inputs], outputs=[bbox, label])
    return model

if __name__ == '__main__':
    model = build_detector(1)

