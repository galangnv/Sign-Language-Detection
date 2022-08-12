import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model

signdetector = load_model('signdetector.h5')
labelMap = {
    0: 'None',
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'D',
    5: 'E',
    6: 'F',
    7: 'G',
    8: 'H',
    9: 'I',
    10: 'J',
    11: 'K',
    12: 'L',
    13: 'M',
    14: 'N',
    15: 'O',
    16: 'P',
    17: 'Q',
    18: 'R',
    19: 'S',
    20: 'T',
    21: 'U',
    22: 'V',
    23: 'W',
    24: 'X',
    25: 'Y',
    26: 'Z'
}

cap = cv2.VideoCapture(1)

while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:500, 50:500,:]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))

    yhat = signdetector.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    labelNum = np.argmax(yhat[0][0])
    label = labelMap[labelNum]

    if np.argmax(yhat[0][0]) != 0:
        cv2.rectangle(
            frame,
            tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
            tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)),
            (255,0,0), 2
        )
        cv2.rectangle(
            frame,
            tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), [0,-30])),
            tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), [80,0])),
            (255,0,0), -1
        )
        cv2.putText(
            frame,
            label,
            tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), [0,-5])),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA
        )

    cv2.imshow('Sign Language Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()