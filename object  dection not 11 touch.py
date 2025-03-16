import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

img = cv2.imread("C:\\Users\\Administrator\\Pictures\\arr2.jpg")
img = cv2.resize(img, (600, 700))

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(img1, 5)
ret, thresh = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY_INV)
cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the threshold value on the "thresh" image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(thresh, f'Threshold: {ret}', (10, 30), font, 1, (0, 0, 255), 2)

cv2.drawContours(img, cnts, -1, (50, 50, 150), 2)
print("Number of contours:", len(cnts))
print("Hierarchy:\n", hier)

# Create a scatter plot for contour widths and heights
widths = [w for x, y, w, h in map(cv2.boundingRect, cnts)]
heights = [h for x, y, w, h in map(cv2.boundingRect, cnts)]

plt.scatter(widths, heights, color='blue')
plt.title('Scatter Plot of Contour Widths and Heights')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()

# Create a confusion matrix (example, not directly applicable to your code)
# In a classification scenario, you'd have true labels and predicted labels
# For this example, let's assume two classes (0 and 1)
true_labels = np.random.randint(2, size=len(cnts))
predicted_labels = np.random.randint(2, size=len(cnts))

cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# The rest of your existing code
