import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Your existing code for image processing
img = cv2.imread("C:\\Users\\Administrator\\Pictures\\applew2.png")
img = cv2.resize(img, (600, 700))
# ...

# Assuming you have ground truth and detected masks
ground_truth_mask = cv2.imread("C:\\Users\\Administrator\\Pictures\\17_100.jpg", cv2.IMREAD_GRAYSCALE)
detected_mask = cv2.imread("C:\\Users\\Administrator\\Pictures\\apple2.jpg", cv2.IMREAD_GRAYSCALE)

# Check if the images are loaded successfully
if ground_truth_mask is None or detected_mask is None:
    print("Error: Could not load images. Check the file paths.")
else:
    # Check if the masks have the same dimensions
    if ground_truth_mask.shape != detected_mask.shape:
        print("Error: Ground truth and detected masks have different dimensions.")
    else:
        # Flatten the masks for confusion matrix calculation
        ground_truth_flat = ground_truth_mask.flatten()
        detected_flat = detected_mask.flatten()

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(ground_truth_flat, detected_flat)

        # Display the confusion matrix with color
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Confusion Matrix")
        plt.show()

        # Assuming you have data for a graph
        x_values = [1, 2, 3, 4, 5]
        y_values = [10, 15, 7, 12, 8]

        # Plot the graph
        plt.plot(x_values, y_values, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=8)
        plt.title("Your Graph Title")
        plt.xlabel("X-axis Label")
        plt.ylabel("Y-axis Label")
        plt.grid(True)
        plt.show()

        # Display the images
        cv2.imshow("original", img)
        cv2.imshow("ground_truth_mask", ground_truth_mask)
        cv2.imshow("detected_mask", detected_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # Your existing code for image processing
# img = cv2.imread("C:\\Users\\Administrator\\Pictures\\applew2.png")
# img = cv2.resize(img, (600, 700))

# img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.medianBlur(img1, 5)
# ret, thresh = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY_INV)
# cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Draw the threshold value on the "thresh" image
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(thresh, f'Threshold: {ret}', (10, 30), font, 1, (0, 0, 255), 2)

# cv2.drawContours(img, cnts, -1, (50, 50, 150), 2)
# print("Number of contours:", len(cnts))
# print("Hierarchy:\n", hier)

# for c in cnts:
#     epsilon = 0.0001 * cv2.arcLength(c, True)
#     data = cv2.approxPolyDP(c, epsilon, True)
#     hull = cv2.convexHull(data)
#     cv2.drawContours(img, [c], -1, (50, 50, 150), 2)
#     cv2.drawContours(img, [hull], -1, (0, 255, 0), 2)

# # Draw measurement lines and values
# for c in cnts:
#     x, y, w, h = cv2.boundingRect(c)
#     cv2.line(img, (x, y - 20), (x + w, y - 20), (0, 255, 0), 2)
#     cv2.putText(img, f'Width: {w}', (x, y - 30), font, 0.8, (0, 255, 0), 1)
#     cv2.line(img, (x - 20, y), (x - 20, y + h), (0, 255, 0), 2)
#     cv2.putText(img, f'Height: {h}', (x - 70, y + h // 2), font, 0.8, (0, 255, 0), 1)

# # Display the original image
# cv2.imshow("original", img)

# # Assuming you have ground truth and detected masks
# ground_truth_mask = cv2.imread("C:\\Users\\Administrator\\Pictures\\17_100.jpg", cv2.IMREAD_GRAYSCALE)
# detected_mask = cv2.imread("C:\\Users\\Administrator\\Pictures\\apple2.jpg", cv2.IMREAD_GRAYSCALE)

# # Check if the images are loaded successfully
# if ground_truth_mask is not None and detected_mask is not None:
#     # Resize the images to the same dimensions if necessary
#     if ground_truth_mask.shape != detected_mask.shape:
#         print("Resizing images to the same dimensions.")
#         new_shape = (max(ground_truth_mask.shape[0], detected_mask.shape[0]),
#                      max(ground_truth_mask.shape[1], detected_mask.shape[1]))

#         ground_truth_mask = cv2.resize(ground_truth_mask, new_shape)
#         detected_mask = cv2.resize(detected_mask, new_shape)

#     # Flatten the masks for confusion matrix calculation
#     ground_truth_flat = ground_truth_mask.flatten()
#     detected_flat = detected_mask.flatten()

#     # Calculate the confusion matrix
#     conf_matrix = confusion_matrix(ground_truth_flat, detected_flat)

#     # Display the confusion matrix with color
#     disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
#     disp.plot(cmap='Blues', values_format='d')
#     plt.title("Confusion Matrix")
#     plt.show()

# # Display the processed images
# cv2.imshow("gray", img1)
# cv2.imshow("thresh", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
