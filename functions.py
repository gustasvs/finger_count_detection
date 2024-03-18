import tensorflow as tf
def preprocess_image(img, target_size=(700, 700), gray_scale=False, single=False):
    images = []
    
    original_shape = tf.cast(tf.shape(img)[:2], tf.float32)
    min_edge = tf.reduce_min(original_shape)
    scale_factor = target_size[0] / min_edge
    new_shape = tf.cast(original_shape * scale_factor, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = tf.image.resize_with_crop_or_pad(img, target_size[0], target_size[1])

    if gray_scale:
        img = tf.image.rgb_to_grayscale(img)
    
    img = img / 255.0
    
    if single:
        return img
    # return [img]
    for k in range(4):
        rotated_img = tf.image.rot90(img, k)
        images.append(rotated_img)

    return images