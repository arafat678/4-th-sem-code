import cv2
import numpy as np

# Load the image
image = cv2.imread("C:\\Users\\Administrator\\Pictures\\colour apple.webp")

# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for yellow, green, and red colors
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# Create masks to isolate yellow, green, and red colors
mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

# Combine masks for red since it wraps around the hue scale
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# Find contours in the color masks
contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a function to label and display apples for each color
def label_and_display_apples(contours, color_name):
    color_image = image.copy()
    count = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 255, 255), 2)  # Draw a white rectangle around the apple
        count += 1

    # Display the color name with "=>" below the apples
    cv2.putText(color_image, f"{color_name} =>", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Display color name with "=>" in black

    cv2.imshow(f"{color_name} Apples", color_image)

# Call the function for each color
label_and_display_apples(contours_yellow, "Yellow")
label_and_display_apples(contours_green, "Green")
label_and_display_apples(contours_red, "Red")

# Wait for user input to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()