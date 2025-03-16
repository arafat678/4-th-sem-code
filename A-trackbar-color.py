import cv2
import numpy as np

frame = cv2.imread("C:\\Users\\Administrator\\Pictures\\17_100.jpg")
frame = cv2.resize(frame, (300, 400))

def nothing(x):
    pass

cv2.namedWindow("color adjust")
cv2.createTrackbar("lowers-h", "color adjust", 0, 255, nothing)
cv2.createTrackbar("lowers-s", "color adjust", 0, 255, nothing)
cv2.createTrackbar("lowers-v", "color adjust", 0, 255, nothing)
cv2.createTrackbar("upers-h", "color adjust", 255, 255, nothing)
cv2.createTrackbar("upers-s", "color adjust", 255, 255, nothing)
cv2.createTrackbar("upers-v", "color adjust", 255, 255, nothing)

while True:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("lowers-h", "color adjust")
    l_s = cv2.getTrackbarPos("lowers-s", "color adjust")
    l_v = cv2.getTrackbarPos("lowers-v", "color adjust")
    u_h = cv2.getTrackbarPos("upers-h", "color adjust")
    u_s = cv2.getTrackbarPos("upers-s", "color adjust")
    u_v = cv2.getTrackbarPos("upers-v", "color adjust")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("original frame", frame)
    cv2.imshow("mask frame", mask)
    cv2.imshow("result frame", result)

    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.waitKey()
cv2.destroyAllWindows()