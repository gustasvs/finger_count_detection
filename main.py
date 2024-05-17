
# imports
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# helper imports
from hparams import BATCH_SIZE, EPOCHS, image_size, gray_scale
from video_interface import video_interface
from functions import preprocess_image, preprocess_and_save_images, preprocessed_image_generator
from model import load_model, create_model

# print-options for less clutter
np.set_printoptions(suppress=True, precision=2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


load_not_train = False
# load_not_train = True

if load_not_train:
    model = load_model('model.h5')
    video_interface(model, image_size)
    sys.exit()


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
model = create_model(input_shape)


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