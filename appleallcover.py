import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image
img = cv2.imread("C:\\Users\\Administrator\\Pictures\\applew2.png")
img = cv2.resize(img, (600, 700))

# Convert the image to grayscale
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply median blur and threshold to create a binary image
blur = cv2.medianBlur(img1, 5)
ret, thresh = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY_INV)

# Find contours
cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and convex hulls on the original image
for c in cnts:
    epsilon = 0.0001 * cv2.arcLength(c, True)
    data = cv2.approxPolyDP(c, epsilon, True)
    hull = cv2.convexHull(data)
    cv2.drawContours(img, [c], -1, (50, 50, 150), 2)
    cv2.drawContours(img, [hull], -1, (0, 255, 0), 2)

# Draw measurement lines and values
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.line(img, (x, y - 20), (x + w, y - 20), (0, 255, 0), 2)
    cv2.putText(img, f'Width: {w}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    cv2.line(img, (x - 20, y), (x - 20, y + h), (0, 255, 0), 2)
    cv2.putText(img, f'Height: {h}', (x - 70, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

# Display the original image with contours
cv2.imshow("original", img)

# Display the grayscale and threshold images
cv2.imshow("gray", img1)
cv2.imshow("thresh", thresh)

# Plot a color histogram
color_channels = cv2.split(img)
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.title('Color Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

