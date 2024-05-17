import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from hparams import *
from functions import preprocess_image

class MovingAverage:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.predictions = []

    def update(self, new_prediction):
        if len(self.predictions) >= self.window_size:
            self.predictions.pop(0)
        self.predictions.append(new_prediction)

    def get_average(self):
        if not self.predictions:
            return None
        return np.mean(self.predictions, axis=0)


def softmax(x):
    res = []
    # exponentionally scale each
    for i in x:
        res.append(i * i)
    res_max = max(res)
    for i in range(len(res)):
        res[i] = res[i] / res_max
    return res

def calibrate_background(cap, num_frames=50):
    """ Capture the background frame """
    background = None
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Smooth to improve mask quality
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if background is None:
            background = gray.copy().astype("float")
            continue
        # Accumulate the background
        cv2.accumulateWeighted(gray, background, 0.5)
    return background

def remove_background(frame, background, threshold=25):
    """ Remove the background from the frame using the calibrated background """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # Compute the difference between the background and current frame
    diff = cv2.absdiff(cv2.convertScaleAbs(background), gray)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def video_interface(model, image_size):
    # 0 for primary cam
    cap = cv2.VideoCapture(0)

    background = calibrate_background(cap)

    # plot to display predictions
    fig, ax = plt.subplots()
    bars = ax.bar(range(1, 6), np.zeros(5), color='gray', width=0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(range(1, 6))
    ax.set_yticklabels([])
    ax.set_yticks([])
    plt.show(block=False)
    
    prediction_smoothing = MovingAverage(window_size=15)
    
    try:
        while True:
            # read frame from the camera
            ret, frame = cap.read()

            if not ret:
                continue

            processed_frame = preprocess_image(frame, (image_size, image_size), gray_scale, single=True)
            
            

            prediction = model.predict(
                np.expand_dims(processed_frame, axis=0),
                verbose=0)[0]
            # prediction = [0.2, np.random.randint(0, 2), 0.2, 0.2, np.random.randint(1, 5)]

            print(prediction)

            prediction = softmax(prediction)

            prediction_smoothing.update(prediction)
            prediction = prediction_smoothing.get_average()

            highest_pred_index = np.argmax(prediction)
            for i, (bar, pred) in enumerate(zip(bars, prediction)):
                bar.set_height(pred)
                bar.set_color('red' if i == highest_pred_index else 'black')
            
            # redraw the plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            

            if isinstance(processed_frame, tf.Tensor):
                processed_frame = processed_frame.numpy()  # Convert to NumPy array
            if processed_frame.dtype != np.uint8:
                processed_frame = (processed_frame * 255).astype(np.uint8)  # Adjust data type and range
            
            cv2.imshow('Cam', processed_frame)

    except Exception as e:
        print(e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close()

if __name__ == '__main__':
    video_interface(None, image_size)
    print("Video Interface Closed (video_inference.py)")