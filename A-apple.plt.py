import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread("C:/Users/Administrator/Pictures/17_100.jpg")

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found.")
else:
    # Calculate histograms for each channel
    hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist3 = cv2.calcHist([img], [2], None, [256], [0, 256])

    # Create a single subplot for the image and histograms
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))

    # Display the original image
    ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title("Original Image")

    # Display the blue histogram
    ax[0, 1].plot(hist1, color='b')
    ax[0, 1].set_title("Blue Histogram")

    # Display the green histogram
    ax[1, 0].plot(hist2, color='g')
    ax[1, 0].set_title("Green Histogram")

    # Display the red histogram
    ax[1, 1].plot(hist3, color='r')
    ax[1, 1].set_title("Red Histogram")

    # Set the x-axis limit for the histograms
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlim([0, 256])

    plt.tight_layout()
    plt.show()