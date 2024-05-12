
# imports
import tensorflow as tf
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

# helper imports
from hparams import BATCH_SIZE, EPOCHS, LR, image_size, gray_scale
from video_interface import video_interface
from functions import preprocess_image, preprocess_and_save_images, preprocessed_image_generator

# print-options for less clutter
np.set_printoptions(suppress=True, precision=2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

input_shape = (image_size, image_size, 1 if gray_scale else 3)

# Load datasets
prprocessed_train_len = preprocess_and_save_images('rokas_train', 'rokas_train_preprocessed', target_size=(image_size, image_size), gray_scale=gray_scale)
steps_per_epoch = prprocessed_train_len // BATCH_SIZE
print("Preprocessed train images:", prprocessed_train_len)

processed_validation_len = preprocess_and_save_images('rokas_validate', 'rokas_validate_preprocessed', target_size=(image_size, image_size), gray_scale=gray_scale)
validation_steps = processed_validation_len // BATCH_SIZE
print("Preprocessed validation images:", processed_validation_len)

train_dataset = tf.data.Dataset.from_generator(
    lambda: preprocessed_image_generator('rokas_train_preprocessed', batch_size=BATCH_SIZE),
    output_types=(tf.float32, tf.float32),
    output_shapes=((None, image_size, image_size, 3), (None, 5))  
)
validation_dataset = tf.data.Dataset.from_generator(
    lambda: preprocessed_image_generator('rokas_validate_preprocessed', batch_size=BATCH_SIZE),
    output_types=(tf.float32, tf.float32),
    output_shapes=((None, image_size, image_size, 3), (None, 5))
)

# model
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score

from keras.applications.mobilenet_v2 import MobileNetV2


base_pretrained_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
base_pretrained_model.trainable = False


model = Sequential([
    base_pretrained_model,
    GlobalAveragePooling2D(),
    
    # # 1st conv layer
    # Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    # MaxPooling2D(pool_size=(2, 2)),
    # # more conv layers
    # Conv2D(128, kernel_size=(3, 3), activation='relu'),
    # MaxPooling2D(pool_size=(2, 2)),
    # Conv2D(256, kernel_size=(3, 3), activation='relu'),
    # MaxPooling2D(pool_size=(2, 2)),
    
    # flatten for dense layers
    Flatten(),
    
    # dense layers for classification
    Dense(256, activation='relu'),
    Dropout(0.5),
    # Dense(64, activation='relu'),
    # Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(
              optimizer=Adam(learning_rate=LR, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
              loss='categorical_crossentropy',
              metrics=['accuracy', F1Score(num_classes=5, average='macro')])


# train 
history = model.fit(train_dataset, 
                    epochs=EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_dataset,
                    validation_steps=validation_steps)

# plot the training history
if 'loss' in history.history:
    plt.plot(history.history['loss'], label='train loss')
if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'], label='train accuracy')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='val loss')
if 'val_f1_score' in history.history:
    plt.plot(history.history['val_f1_score'], label='val F1 Score')

plt.title('Training History')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# save the model
model.save('model.h5')


video_interface(model, image_size)