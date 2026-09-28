import cv2
import os

def save(image, filename):

    os.makedirs("output", exist_ok=True)

    cv2.imwrite(
        os.path.join("output", filename),
        image
    )