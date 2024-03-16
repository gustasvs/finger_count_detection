import tensorflow as tf
import os
import numpy as np
import random

# hparams
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.01

# printoptions for easier debugging
np.set_printoptions(suppress=True, precision=2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print_debug_info = True

def preprocess_image(img, target_size=(700, 700)):
    img = tf.image.resize_with_pad(img, target_size[0], target_size[1])
    # img = tf.image.resize(img, [target_size[0], target_size[0]], preserve_aspect_ratio=True)
    img = tf.image.central_crop(img, central_fraction=1.0)
    # black and white image
    # img = tf.image.rgb_to_grayscale(img)
    return img


# Function to read images and extract labels
def laod_images(folder_path, target_size=(700, 700)):
    filenames = os.listdir(folder_path)
    images = []
    labels = []

    for filename in filenames:
        label = int(filename.split('_')[0])
        if 1 <= label <= 5:
            img_path = os.path.join(folder_path, filename)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)  # Ensure image has three color channels
            img = preprocess_image(img, target_size)
            images.append(img)
            labels.append(label)
    
    return tf.data.Dataset.from_tensor_slices((images, labels))


image_size = 50
# Load datasets
train_dataset = laod_images('rokas_train', target_size=(image_size, image_size))
validate_dataset = laod_images('rokas_validate', target_size=(image_size, image_size))

# count labels
def count_labels(dataset):
    label_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for _, label in dataset.as_numpy_iterator():
        if label in label_counts:
            label_counts[label] += 1
    return label_counts

print("Train label counts:", count_labels(train_dataset))
print("Validate label counts:", count_labels(validate_dataset))


# model
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam

model = Sequential([
    # input layer
    Conv2D(32, activation='relu', kernel_size=(3, 3), input_shape=(image_size, image_size, 3)),
    BatchNormalization(),

    # convolutional layers
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    # output layer
    Flatten(),
    Dense(5, activation='softmax')
    ])

optimizer = Adam(learning_rate=LR, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss='mse')

# simple model for testing that returns random image
def model_predict_test():
    return random.choice([1, 2, 3, 4, 5])

model.fit(train_dataset.batch(BATCH_SIZE), epochs=EPOCHS, validation_data=validate_dataset.batch(BATCH_SIZE))

#