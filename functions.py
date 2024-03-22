import tensorflow as tf
def preprocess_image(img, target_size=(700, 700), gray_scale=False, single=False):
    images = []
    
    original_shape = tf.cast(tf.shape(img)[:2], tf.float32)
    min_edge = tf.reduce_min(original_shape)
    # to counter rounding being unprecise
    scale_factor = target_size[0] * 1.001 / min_edge
    new_shape = tf.cast(tf.round(original_shape * scale_factor), tf.int32)
    img = tf.image.resize(img, new_shape)
    # print(img.shape)
    if gray_scale:
        img = tf.image.rgb_to_grayscale(img)
    img = img / 255.0
    
    if single:
        img = tf.image.resize_with_crop_or_pad(img, target_size[0], target_size[1])
        return img
    
    shape = tf.shape(img)
    if shape[0] < shape[1]: # wider
        for i in range(3):
            start = (shape[1] - target_size[1]) // 2 * i
            cropped_img = tf.image.crop_to_bounding_box(img, 0, start, target_size[0], target_size[1])
            images.append(cropped_img)
    else: # taller
        for i in range(3):
            start = (shape[0] - target_size[0]) // 2 * i
            cropped_img = tf.image.crop_to_bounding_box(img, start, 0, target_size[0], target_size[1])
            images.append(cropped_img)
    
    cropped_images = []
    
    for img in images:
        # cropped_images.append(img)
        for k in range(4):
            rotated_img = tf.image.rot90(img, k)
            cropped_images.append(rotated_img)

    return cropped_images