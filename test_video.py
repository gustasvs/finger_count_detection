import cv2
import numpy as np

def get_frame_difference(prev_frame, current_frame):
    """ Calculate the absolute difference between two frames """
    diff = cv2.absdiff(prev_frame, current_frame)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return thresh

def preprocess_frame(frame):
    """ Convert frame to grayscale and apply Gaussian Blur """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray

from PIL import Image
import io


def main():
    cap = cv2.VideoCapture(0)
    ret, prev_frame = cap.read()
    prev_frame = preprocess_frame(prev_frame)

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        current_frame_processed = preprocess_frame(current_frame_rgb)
        # current_frame_processed = remove_bg(current_frame_processed)

        motion_mask = get_frame_difference(prev_frame, current_frame_processed)

        # Update previous frame
        prev_frame = current_frame_processed

        cv2.imshow('Motion Mask', motion_mask)
        cv2.imshow('Current Frame', current_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
