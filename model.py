# model
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score

from keras.applications.mobilenet_v2 import MobileNetV2

from hparams import LR

def load_model(filename):
    return tf.keras.models.load_model(filename, custom_objects={'F1Score': F1Score(num_classes=5, average='macro')})

def create_model(input_shape):

    # base_pretrained_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    # base_pretrained_model.trainable = False

    model = Sequential([
        # base_pretrained_model,
        # GlobalAveragePooling2D(),
        
        # conv layers
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # flatten for dense layers
        Flatten(),
        Dropout(0.5),
        
        # dense layers for classification
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(5, activation='softmax')
    ])

    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LR,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )

    model.compile(
                optimizer=Adam(learning_rate=LR, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                loss='categorical_crossentropy',
                metrics=['accuracy', F1Score(num_classes=5, average='macro')])
    
    return model