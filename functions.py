import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt

def preprocess_image(img, target_size=(700, 700), gray_scale=False, single=False):    
    original_shape = tf.cast(tf.shape(img)[:2], tf.float32)
    min_edge = tf.reduce_min(original_shape)
    # fix to counter rounding being unprecise
    scale_factor = target_size[0] * 1.001 / min_edge
    new_shape = tf.cast(tf.round(original_shape * scale_factor), tf.int32)
    img = tf.image.resize(img, new_shape)
    if gray_scale:
        img = tf.image.rgb_to_grayscale(img)
    img = img / 255.0
    
    # handle case when we need to return single image
    if single:
        img = tf.image.resize_with_crop_or_pad(img, target_size[0], target_size[1])
        return img
    
    # step 1 crop images
    cropped_images = []
    shape = tf.shape(img)
     # wider
    if shape[0] < shape[1]:
        for i in range(3):
            start = (shape[1] - target_size[1]) // 2 * i
            cropped_img = tf.image.crop_to_bounding_box(img, 0, start, target_size[0], target_size[1])
            cropped_images.append(cropped_img)
    # taller
    else: 
        for i in range(3):
            start = (shape[0] - target_size[0]) // 2 * i
            cropped_img = tf.image.crop_to_bounding_box(img, start, 0, target_size[0], target_size[1])
            cropped_images.append(cropped_img)

    # take out random 1 from the cropped images
    if len(cropped_images) > 2:
        cropped_images.pop(random.randint(0, len(cropped_images) - 1))


    # step 2 color jitter images
    color_jittered_images = []
    for img in cropped_images:
        # original image so it's always there
        color_jittered_images.append(img)
        for _ in range(2):
            color_jittered_img = tf.image.random_brightness(img, max_delta=0.2)
            color_jittered_img = tf.image.random_contrast(color_jittered_img, lower=0.1, upper=0.3)
            color_jittered_images.append(color_jittered_img)

    # step 3 rotate images
    
    rotated_images = []
    for img in color_jittered_images:
        rotated_images.append(img)
        rotated_images.append(tf.image.rot90(img, k=1)) # left
        rotated_images.append(tf.image.rot90(img, k=3)) # right

    return rotated_images


def preprocess_and_save_images(input_folder, output_folder, target_size=(700, 700), gray_scale=False):
    filenames = os.listdir(input_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    elif len(os.listdir(output_folder)) > 0:
        return len(os.listdir(output_folder))
        for filename in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, filename))
        
    

    filenames = os.listdir(input_folder)
    for filename in filenames:
        with open(os.path.join(input_folder, filename), 'rb') as f:
            img = tf.image.decode_jpeg(f.read())
            preprocessed_images = preprocess_image(img, target_size, gray_scale)
            for i, preprocessed_img in enumerate(preprocessed_images):
                if gray_scale:
                    preprocessed_img = tf.image.grayscale_to_rgb(preprocessed_img)
                # plt.imshow(preprocessed_img)
                # plt.show()
                with open(os.path.join(output_folder, f'{filename.split(".")[0]}_{i}.jpg'), 'wb') as f:
                    f.write(tf.image.encode_jpeg(tf.cast(preprocessed_img * 255, tf.uint8)).numpy())
    return len(os.listdir(output_folder))

def preprocessed_image_generator(folder_path, batch_size=32):
    while True:  # Loop indefinitely
        image_files = os.listdir(folder_path)  # List all files in the folder

        
        labels = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        label_groups = {1: [], 2: [], 3: [], 4: [], 5: []}
        for filename in image_files:
            label = int(filename.split('_')[0])
            labels[label] += 1
            label_groups[label].append(filename)

        min_label = min(labels.values())

        print(labels)

        # Normalize all groups to the same size (min_label) by undersampling
        normalized_files = []
        for label, files in label_groups.items():
            if len(files) > min_label:
                files = random.sample(files, min_label)
            normalized_files.extend(files)

        # Shuffle to randomize the order for each epoch
        random.shuffle(normalized_files)
    

        batch_images = []
        batch_labels = []

        for filename in normalized_files:
            image_path = os.path.join(folder_path, filename)
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)  # or 1 for grayscale
            img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize the image to [0, 1]

            # Extract label from filename or separate file
            label = extract_label_from_filename(filename)  # Implement this function based on your labeling

            batch_images.append(img)
            batch_labels.append(label)

            # show iamge and print label
            # print(label)
            # plt.imshow(img)
            # plt.show()

            # Check if the batch is full
            if len(batch_images) >= batch_size:
                yield np.stack(batch_images, axis=0), np.stack(batch_labels, axis=0)
                batch_images, batch_labels = [], []

        # If there's any leftover data less than a full batch, yield it as well
        if batch_images:
            yield np.stack(batch_images, axis=0), np.stack(batch_labels, axis=0)

def extract_label_from_filename(filename):
    # example filename 3_92_2.jpg
    label_part = filename.split('_')[0]
    label = int(label_part)
    
    one_hot_label = tf.one_hot(label - 1, depth=5)

    return one_hot_label
    







import tensorflow as tf
import os

def test_preprocessed_image_generator():
    # Path to the directory containing preprocessed images
    folder_path = 'rokas_train_preprocessed'
    batch_size = 32  # or any suitable batch size that matches your BATCH_SIZE

    # Create the generator
    gen = preprocessed_image_generator(folder_path, batch_size=batch_size)

    # Get the first batch
    images, labels = next(gen)

    # Print shapes and some example data
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("First image min and max:", images[0].min(), images[0].max())
    print("First label:", labels[0])

    # If you want to visually inspect an image
    import matplotlib.pyplot as plt
    plt.imshow(images[0], cmap='gray' if images[0].shape[-1] == 1 else None)
    plt.title("Sample Image from Generator")
    plt.show()

# Ensure the folder exists and contains images before running this test
# if os.path.exists('rokas_train_preprocessed'):
#     test_preprocessed_image_generator()
# else:
#     print("The folder is not set up correctly or is empty.")
