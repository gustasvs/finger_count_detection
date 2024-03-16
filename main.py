import tensorflow as tf
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

# printoptions for easier debugging
np.set_printoptions(suppress=True, precision=2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print_debug_info = True

# hparams
BATCH_SIZE = 6
EPOCHS = 15
LR = 0.01

# image params
image_size = 80
gray_scale = False


def preprocess_image(img, target_size=(700, 700)):
    # Determine the scale factor to make the smallest edge equal to the target size
    original_shape = tf.cast(tf.shape(img)[:2], tf.float32)
    min_edge = tf.reduce_min(original_shape)
    scale_factor = target_size[0] / min_edge
    
    # Calculate the new shape after scaling
    new_shape = tf.cast(original_shape * scale_factor, tf.int32)
    
    # Resize the image according to the new shape
    img = tf.image.resize(img, new_shape)
    
    # Crop the central part of the resized image to make it a square of target_size x target_size
    img = tf.image.resize_with_crop_or_pad(img, target_size[0], target_size[1])
    


    if gray_scale:
        img = tf.image.rgb_to_grayscale(img)
    # print("Image shape:", img.shape)
    # print("Before normalization:", "Min:", tf.reduce_min(img).numpy(), "Max:", tf.reduce_max(img).numpy())
    # Normalize the image to [0, 1]
    img = img / 255.0
    # print("After normalization:", "Min:", tf.reduce_min(img).numpy(), "Max:", tf.reduce_max(img).numpy())
    # sys.exit(0)
    # plt.imshow(img.numpy().astype('uint8'), cmap='gray')
    # plt.imshow(img.numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    # plt.colorbar() 
    # plt.show()
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
            img = tf.image.decode_jpeg(img, channels=3) 
            img = preprocess_image(img, target_size)
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

base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    # Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    # MaxPooling2D(pool_size=(2, 2)),
    # Dropout(0.25),
    base_model,
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


import cv2
import numpy as np
import matplotlib.pyplot as plt

# 0 for primary cam
cap = cv2.VideoCapture(0)

fig, ax = plt.subplots()
bars = ax.bar(range(1, 6), np.zeros(5), color='blue', width=0.5)
ax.set_ylim(0, 1)
plt.ion() # interactive mode on
plt.show()

def softmax(x):
    res = []
    # exponentionally scale each
    for i in x:
        res.append(i * i)
    res_max = max(res)
    for i in range(len(res)):
        res[i] = res[i] / res_max
    return res

while True:
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if no frame is captured
    # model_prediction = model.predict(preprocess_image(frame, (image_size, image_size))), batch_size=1)[0]
    # prediction = np.argmax(model_prediction) + 1
    prediction = model.predict(
        np.expand_dims(
            preprocess_image(frame, (image_size, image_size)), 
            axis=0), verbose=0)[0]
    
    prediction = softmax(prediction)

    highest_pred_index = np.argmax(prediction)
    
    for i, (bar, pred) in enumerate(zip(bars, prediction)):
        bar.set_height(pred)
        
        if i == highest_pred_index:
            bar.set_color('red')
        else:
            bar.set_color('black')
    
    fig.canvas.draw()
    fig.canvas.flush_events()

    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the plot
cap.release()
cv2.destroyAllWindows()
plt.close()


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