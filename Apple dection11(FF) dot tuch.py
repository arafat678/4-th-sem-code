
import cv2

def objectDetection(image_path):
    image_list = []
    img = cv2.imread(image_path)
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
        cv2.putText(img, f'Width: {w}', (x, y - 30), font, 0.8, (0, 255, 0), 1)
        cv2.line(img, (x - 20, y), (x - 20, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'Height: {h}', (x - 70, y + h // 2), font, 0.8, (0, 255, 0), 1)

    # Calculate and display height and width at the top of the image
    total_width = sum([w for x, y, w, h in map(cv2.boundingRect, cnts)])
    total_height = sum([h for x, y, w, h in map(cv2.boundingRect, cnts)])
    cv2.putText(img, f'Total Width: {total_width}', (10, 50), font, 1, (0, 0, 255), 2)
    cv2.putText(img, f'Total Height: {total_height}', (10, 80), font, 1, (0, 0, 255), 2)
    cv2.imwrite('output_image-1.jpg', img)
    image_list.append('output_image-1.jpg')
    cv2.imwrite('output_image-2.jpg', img1)
    image_list.append('output_image-2.jpg')
    cv2.imwrite('output_image-3.jpg', thresh)
    image_list.append('output_image-3.jpg')
    return image_list
    
    
    cv2.imshow("original", img)
    cv2.imshow("gray", img1)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
myImage = "C:/Users/Administrator/Pictures/applew2.png"
data = objectDetection(myImage)

print(data)