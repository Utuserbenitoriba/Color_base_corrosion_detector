import numpy as np
import argparse
import cv2
from pathlib import Path
import os

# H S V color ranges for corrosion. (Hue, Saturation, Value)
lower_corrosion = np.array([0,50,50])
upper_corrosion = np.array([12,255,255])
# Corrosion color red is split into two ranges in the hsv color space
lower_corrosion2 = np.array([165,50,50])
upper_corrosion2 = np.array([179,255,255])
# Defining a more rigorous range for the corrosion inside the rectangle
lower_corrosion_inside = np.array([0,50,50])
upper_corrosion_inside = np.array([17,255,255])

# HSV ranges for black and dark red colors
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])

lower_dark_red = np.array([0, 50, 50])
upper_dark_red = np.array([10, 255, 100])

lower_brown = np.array([10, 100, 20])
upper_brown = np.array([20, 255, 200])

lower_dark_brown = np.array([10, 50, 20])
upper_dark_brown = np.array([20, 255, 100])


input_dir = Path('corrosion_pictures')
output_dir = Path('processed_pictures')
output_dir.mkdir(exist_ok=True)

for image_path in input_dir.glob('*.png'):
    image = cv2.imread(str(image_path))

    # Check if the image was loaded successfully
    if image is None:
        print("Error: Could not load image {image_path}.")
        continue
    else:
        print(f"Processing image {image_path}...")
        #print("Image shape:", image.shape)  # Show dimensions (height, width, channels)

    # Convert the image to the HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the corrosion color
    mask1 = cv2.inRange(image_hsv, lower_corrosion, upper_corrosion)
    mask2 = cv2.inRange(image_hsv, lower_corrosion2, upper_corrosion2)

    # Combine the two masks to look for corrosion
    mask = cv2.bitwise_or(mask1, mask2)

    # Create a mask for the inside of the rectangle, different colors
    mask_broader = cv2.inRange(image_hsv, lower_corrosion_inside, upper_corrosion_inside)
    mask_inside = cv2.inRange(image_hsv, lower_corrosion_inside, upper_corrosion_inside)
    mask_black = cv2.inRange(image_hsv, lower_black, upper_black)
    mask_dark_red = cv2.inRange(image_hsv, lower_dark_red, upper_dark_red)
    mask_brown = cv2.inRange(image_hsv, lower_brown, upper_brown)
    mask_dark_brown = cv2.inRange(image_hsv, lower_dark_brown, upper_dark_brown)

    # Combine all masks
    mask_inside = cv2.bitwise_or(mask_broader, mask_black)
    mask_inside = cv2.bitwise_or(mask_inside, mask_dark_red)
    mask_inside = cv2.bitwise_or(mask_inside, mask_brown)
    mask_inside = cv2.bitwise_or(mask_inside, mask_dark_brown)


    # Ignoring a rectangle in the image (date of in the picture)
    x1, y1 = 600, 510
    x3, y3 = 800, 550
    mask[y1:y3, x1:x3] = 0

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_colored = np.zeros_like(image)  # Initialize a colored mask with zeros

    if len(contours) != 0:
        for contour in contours:
            if cv2.contourArea(contour) > 30:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Change the color of the mask inside the rectangle to green
                mask_colored[y:y+h, x:x+w][mask_inside[y:y+h, x:x+w] == 255] = [0, 255, 0]

                # Label the rectangle
                text = "Rust Zone"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (0, 255, 0)
                thickness = 1
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = x
                text_y = y + h + text_size[1] + 5  # 5 pixels below the rectangle
                cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, thickness)


    # Save the result image in the output directory
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), image)

