import tensorflow as tf
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

from hparams import BATCH_SIZE, EPOCHS, LR, image_size, gray_scale

from video_interface import video_interface

# printoptions for easier debugging
np.set_printoptions(suppress=True, precision=2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print_debug_info = True


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
            img = tf.image.decode_jpeg(img, channels=3) 
            img = preprocess_image(img, target_size, gray_scale=gray_scale)
            images.append(img)
            labels.append(np.eye(5)[label - 1])
            # print(np.eye(5)[label - 1])
    
    return tf.data.Dataset.from_tensor_slices((images, labels))

# Load datasets
train_dataset = laod_images('rokas_train', target_size=(image_size, image_size))
validate_dataset = laod_images('rokas_validate', target_size=(image_size, image_size))

train_dataset = train_dataset.shuffle(buffer_size=600)

# print one example
# for img, label in train_dataset.take(1):
#     print("Image shape:", img.shape)
#     print("Label:", label)
#     # plt.imshow(img.numpy().astype('uint8'), cmap='gray')
#     plt.title(f"Label: {label}")
#     plt.axis('off')
#     plt.imshow(img.numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
#     plt.colorbar() 
#     plt.show()

# count labels function
def count_labels(dataset):
    label_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for _, label in dataset.as_numpy_iterator():
        label = np.argmax(label) + 1
        label_counts[label] += 1
    return label_counts

print("Train label counts:", count_labels(train_dataset))
print("Validate label counts:", count_labels(validate_dataset))


# model
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score

from keras.applications.mobilenet_v2 import MobileNetV2

input_shape = (image_size, image_size, 1 if gray_scale else 3)

base_pretrained_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
base_pretrained_model.trainable = False

model = Sequential([
    # Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    # MaxPooling2D(pool_size=(2, 2)),
    # Dropout(0.25),
    base_pretrained_model,
    GlobalAveragePooling2D(),

    # Conv2D(64, (3, 3), activation='relu'),
    # MaxPooling2D(pool_size=(2, 2)),
    # Dropout(0.25),

    # Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

optimizer = Adam(learning_rate=LR, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer="adam", 
              loss='categorical_crossentropy',
              metrics=['accuracy', F1Score(num_classes=5, average='macro')])


# simple model for testing that returns random image
def model_predict_test():
    return random.choice([1, 2, 3, 4, 5])

history = model.fit(train_dataset.batch(BATCH_SIZE), 
                    epochs=EPOCHS, validation_data=validate_dataset.batch(BATCH_SIZE))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

video_interface(model, image_size)

# test the model, make a prediction and display the image
def test_model():
    try:
        for img, label in validate_dataset.shuffle(buffer_size=200):
            model_prediction = model.predict(np.array([img]), batch_size=1)[0]
            print("Model prediction:", model_prediction)
            prediction = np.argmax(model_prediction) + 1

            prediction = model_prediction

            img = img.numpy()
            label = label.numpy()
            print("Real count:", label)
            print("Predicted count:", prediction)
            print("Image shape:", img.shape)
            print("Image:")

            plt.figure(figsize=(4, 5))
            # plt.imshow(img.astype('uint8'), cmap='gray')
            plt.title(f"real: {label} \npred: {prediction}")
            plt.axis('off')
            plt.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
            # plt.colorbar()
            plt.show()
    except KeyboardInterrupt:
        print("Test ended")

test_model()