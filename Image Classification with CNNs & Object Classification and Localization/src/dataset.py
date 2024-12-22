import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import to_categorical
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_image(image_path: str, label):
    def decode_image_path(image_path_tensor):
        return image_path_tensor.numpy().decode("utf-8")

    image_path_str = tf.py_function(decode_image_path, [image_path], tf.string)

    def read_image(path):
        with Image.open(path.numpy()) as image:
            return image.resize((128, 128)).convert("RGB")

    image = tf.py_function(read_image, [image_path_str], tf.uint8)
    image.set_shape((128, 128, 3))
    
    # Rescale pixel values to the range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_dataset(txt_path, batch_size, data_dir):
    with open(txt_path, "r") as f:
        image_paths = f.read().splitlines()

    image_paths = [os.path.join(data_dir, path) for path in image_paths]
    labels = [path.split("/")[-2] for path in image_paths]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(500)
    dataset = dataset.batch(batch_size)
    
    return dataset

def clean_data(data_dir):
    labels = os.listdir(data_dir)
    lengths = {label:len(os.listdir(os.path.join(data_dir, label))) for label in labels}
    cleaned_labels = []
    for label in lengths:
        if lengths[label] > 250:
            cleaned_labels.append(label)

    return cleaned_labels[:15]
def create_paths(data_dir, labels):
    train_paths, test_paths = [], []
    for label in labels:
        paths = []
        for path in os.listdir(os.path.join(data_dir, label)):
            path = os.path.join(label, path)
            paths.append(path)

        train, test = train_test_split(paths, test_size=0.2, random_state=42)
        train_paths.extend(train)
        test_paths.extend(test)
    
    return train_paths, test_paths

def write_paths(train_paths, test_paths, train_txt_path, test_txt_path):
    with open(train_txt_path, 'w') as f:
        f.write('\n'.join(train_paths))

    with open(test_txt_path, 'w') as f:
        f.write('\n'.join(test_paths))
        
def preprocess_image_detection(image_path, coord, label):
    def decode_image_path(image_path_tensor):
        return image_path_tensor.numpy().decode("utf-8")

    image_path_str = tf.py_function(decode_image_path, [image_path], tf.string)

    def read_image(path):
        with Image.open(path.numpy()) as image:
            return image.resize((128, 128)).convert("RGB")

    image = tf.py_function(read_image, [image_path_str], tf.uint8)
    image.set_shape((128, 128, 3))
    
    # Rescale pixel values to the range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, coord, label

def detection_dataset(annotations_txt, image_folder, image_size, batch_size):
    image_paths = []
    coordinates = []
    labels = []

    with open(annotations_txt, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        image_path, *rest = line.split(' ')
        image_path = os.path.join(image_folder, image_path)
        image_paths.append(image_path)
        for xmin, ymin, xmax, ymax, label in rest:
            xmin /= image_size
            ymin /= image_size
            xmax /= image_size
            ymax /= image_size
            coordinates.append((xmin, ymin, xmax, ymax))
            labels.append(label)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, coordinates, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(500)
    dataset = dataset.batch(batch_size)

if __name__ == '__main__':
    DATADIR = 'src/data/indoorCVPR_09/Images'
    TRAIN_IMAGE_PATH = 'src/data/TrainImages.txt'
    TEST_IMAGE_PATH = 'src/data/TestImages.txt'
    labels = clean_data(DATADIR)
    train_paths, test_paths = create_paths(DATADIR, labels)
    write_paths(train_paths, test_paths, TRAIN_IMAGE_PATH, TEST_IMAGE_PATH)