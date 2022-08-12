import os
import time
import uuid
import cv2

label = 'sign_a'    # Change to which sign you're collecting
IMAGES_PATH = os.path.join('data', 'images', label)
numImagesPerLabel = 15

if not os.path.exists(IMAGES_PATH):
    os.mkdir(IMAGES_PATH)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.imshow('Starting...', frame)
    cv2.moveWindow('Starting...', 0, 0)

    if cv2.waitKey(1) & 0xFF == ord('g'):
        break
cv2.destroyAllWindows()

for imgNum in range(1, numImagesPerLabel + 1):
    print(f'Collecting image {imgNum} for label {label}')
    _, frame = cap.read()
    cv2.imshow(f'Collecting {label} images', frame)
    cv2.moveWindow(f'Collecting {label} images', 0, 0)
    time.sleep(1)
    imgName = os.path.join(IMAGES_PATH, f'sign-{label.split("_")[1]}-{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgName, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()