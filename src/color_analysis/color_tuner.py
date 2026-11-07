"""
Interactive HSV Color Space Tuner
Provides real-time OpenCV trackbar calibration for fruit defect and surface segmentation.
"""

import cv2
import numpy as np


def nothing(x):
    pass


def tune_hsv_trackbars(image_path: str, window_name: str = "HSV Calibrator") -> None:
    """
    Launches an interactive OpenCV window with 6 trackbars for tuning
    lower and upper HSV bounds on fruit images.
    Press 'ESC' or 'q' to quit.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    frame = cv2.resize(frame, (400, 500))

    cv2.namedWindow(window_name)
    cv2.createTrackbar("lower_h", window_name, 0, 180, nothing)
    cv2.createTrackbar("lower_s", window_name, 50, 255, nothing)
    cv2.createTrackbar("lower_v", window_name, 50, 255, nothing)
    cv2.createTrackbar("upper_h", window_name, 180, 180, nothing)
    cv2.createTrackbar("upper_s", window_name, 255, 255, nothing)
    cv2.createTrackbar("upper_v", window_name, 255, 255, nothing)

    print("[INFO] Press 'q' or 'ESC' to exit tuner.")

    while True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lh = cv2.getTrackbarPos("lower_h", window_name)
        ls = cv2.getTrackbarPos("lower_s", window_name)
        lv = cv2.getTrackbarPos("lower_v", window_name)
        uh = cv2.getTrackbarPos("upper_h", window_name)
        us = cv2.getTrackbarPos("upper_s", window_name)
        uv = cv2.getTrackbarPos("upper_v", window_name)

        lower_bound = np.array([lh, ls, lv], dtype=np.uint8)
        upper_bound = np.array([uh, us, uv], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        stacked = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result])
        cv2.imshow(window_name, stacked)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()
