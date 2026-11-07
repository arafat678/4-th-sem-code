"""
Multi-Color Range Segmentation for Apple Varietal Analysis
Segments red, yellow, and green regions using predefined HSV masks.
"""

from typing import Dict, Tuple
import cv2
import numpy as np


COLOR_RANGES = {
    "red_lower": (np.array([0, 70, 50]), np.array([10, 255, 255])),
    "red_upper": (np.array([170, 70, 50]), np.array([180, 255, 255])),
    "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
    "green": (np.array([35, 50, 50]), np.array([85, 255, 255]))
}


def segment_by_color_ranges(
    image: np.ndarray
) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Computes masks and surface coverage percentages for red, yellow, and green colors.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_pixels = float(image.shape[0] * image.shape[1])
    results = {}

    # Red spans both ends of H axis in OpenCV HSV (0-10 and 170-180)
    mask_r1 = cv2.inRange(hsv, COLOR_RANGES["red_lower"][0], COLOR_RANGES["red_lower"][1])
    mask_r2 = cv2.inRange(hsv, COLOR_RANGES["red_upper"][0], COLOR_RANGES["red_upper"][1])
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    red_coverage = (float(np.count_nonzero(mask_red)) / total_pixels) * 100.0
    results["red"] = (mask_red, round(red_coverage, 2))

    # Yellow
    mask_yellow = cv2.inRange(hsv, COLOR_RANGES["yellow"][0], COLOR_RANGES["yellow"][1])
    yellow_coverage = (float(np.count_nonzero(mask_yellow)) / total_pixels) * 100.0
    results["yellow"] = (mask_yellow, round(yellow_coverage, 2))

    # Green
    mask_green = cv2.inRange(hsv, COLOR_RANGES["green"][0], COLOR_RANGES["green"][1])
    green_coverage = (float(np.count_nonzero(mask_green)) / total_pixels) * 100.0
    results["green"] = (mask_green, round(green_coverage, 2))

    return results
