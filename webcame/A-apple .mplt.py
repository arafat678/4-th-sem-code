
import cv2
import matplotlib.pyplot as plt
import numpy as np

nemo = cv2.imread("C:\\Users\\Administrator\\Pictures\\apple2.jpg")
plt.imshow(nemo)
plt.show()
nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
plt.imshow(nemo)
plt.show()