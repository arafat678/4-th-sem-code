import cv2
import numpy as np

img = cv2.imread("C:\\Users\\Administrator\\Pictures\\applew2.png")
img =cv2.resize(img,(600,700))

img1= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(img1,5)
ret,thresh=cv2.threshold(blur,210,255,cv2.THRESH_BINARY_INV)
cnts,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,cnts,-1,(50,50,150),2)
print("number of contours==",cnts,"\ntotal contour==",len(cnts))
print("hierarchy==\n",hier)
for c in cnts:
    epsilon =0.0001*cv2.arcLength(c,True)
    data=cv2.approxPolyDP(c,epsilon,True)
    hull=cv2.convexHull(data)
    cv2.drawContours(img,[c],-1,(50,50,150),2)
    cv2.drawContours(img,[hull],-1,(0,255,0),2)
    
# gray = cv2.medianBlur(gray, 5)
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for (x, y, r) in circles[0, :]:
#         cv2.circle(img2, (x, y), r, (50, 10, 50), 3)
#         cv2.circle(img2, (x, y), 2, (0, 255, 100), -1)

cv2.imshow("original", img)
cv2.imshow("gray",img1)
cv2.imshow("thresh",thresh)
cv2.waitKey(0)
# cv2.destroyAllWindows()