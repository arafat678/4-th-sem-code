import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load your ground truth and detected masks (assuming binary masks)
ground_truth_mask = cv2.imread("C:\\Users\\Administrator\\Pictures\\apple1.png", cv2.IMREAD_GRAYSCALE)
detected_mask = cv2.imread("C:\\Users\\Administrator\\Pictures\\apple1.png", cv2.IMREAD_GRAYSCALE)

# Convert the BGR image to RGB format
img = cv2.imread("C:\\Users\\Administrator\\Pictures\\apple1.png")
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the original image
plt.imshow(RGB_img)
plt.title("Original Image")
plt.show()

# Display the ground truth mask
plt.imshow(ground_truth_mask, cmap='gray')
plt.title("Ground Truth Mask")
plt.show()

# Display the detected mask
plt.imshow(detected_mask, cmap='gray')
plt.title("Detected Mask")
plt.show()

# Flatten the masks for confusion matrix calculation
ground_truth_flat = ground_truth_mask.flatten()
detected_flat = detected_mask.flatten()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(ground_truth_flat, detected_flat, test_size=0.2, random_state=42)

# Create a kNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train.reshape(-1, 1), y_train)

# Predict using kNN
y_pred = knn_classifier.predict(X_test.reshape(-1, 1))

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate accuracy (optional)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Display the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix as an image
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
