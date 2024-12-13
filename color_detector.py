import numpy as np
import argparse
import cv2
from pathlib import Path
a = 14 
# H S V color ranges for corrosion
lower_corrosion = np.array([0,50,50])
upper_corrosion = np.array([11,255,255])

lower_corrosion2 = np.array([175,50,50])
upper_corrosion2 = np.array([179,255,255])

image_path = Path('corrosion_pictures/1.png')
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully!")
    print("Image shape:", image.shape)  # Show dimensions (height, width, channels)

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(image_hsv, lower_corrosion, upper_corrosion)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) != 0:
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)


cv2.imshow('Mask', mask)
cv2.imshow('hsv Image', image_hsv)
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
