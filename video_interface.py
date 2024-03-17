import cv2
import numpy as np
import matplotlib.pyplot as plt
from hparams import *
from functions import preprocess_image

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

def video_interface(model, image_size):
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if no frame is captured
        # model_prediction = model.predict(preprocess_image(frame, (image_size, image_size))), batch_size=1)[0]
        # prediction = np.argmax(model_prediction) + 1
        prediction = model.predict(
            np.expand_dims(
                preprocess_image(frame, (image_size, image_size), gray_scale), 
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