import tensorflow as tf
import numpy as np
import glob
import cv2
import os

from six import BytesIO
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tf.keras.backend.clear_session()
model_path = 'inference_graph/saved_model'
model = tf.saved_model.load(model_path)

def scale_bbox(w, h, bbox):
    y1, x1, y2, x2 = tf.unstack(bbox, axis=-1)
    y1 = y1 * w
    x1 = x1 * h
    y2 = y2 * w
    x2 = x2 * h
    return np.array(tf.stack([y1, x1, y2, x2], axis=1)).astype(np.int32)

def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]

    # inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    y_pred = tf.squeeze(output_dict['detection_multiclass_scores'])
    b_pred = tf.squeeze(output_dict['detection_boxes'])
    
    return y_pred, b_pred

def draw(image, y_pred, b_pred, confidence):
    for i in range(y_pred.shape[0]):
        if y_pred[i] > confidence:
            y1, x1, y2, x2 = b_pred[i]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image

if __name__ == '__main__':
    test_path = 'test_images/*.jpg'
    out_path = 'samples'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    confidence = 0.5
        
    for i, image_path in enumerate(glob.glob(test_path)):
        image_np = load_image_into_numpy_array(image_path)
        w, h, _ = image_np.shape

        y_pred, b_pred = run_inference_for_single_image(model, image_np)
        b_pred = scale_bbox(w, h, b_pred)
        
        out_image = draw(image_np, y_pred, b_pred, confidence)
        out_image = Image.fromarray(np.array(out_image, dtype=np.uint8))
        out_image.save(out_path + '/' + str(i) + '.jpg')

