"""
Physical Trait Extraction and Contour Analysis Module
Computes fruit dimensions (width, height), bounding box, convex hull,
aspect ratio, and surface defect region statistics.
"""

import os
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


def detect_contours_and_hull(
    image: np.ndarray,
    threshold_val: int = 210,
    blur_kernel: int = 5,
    approx_epsilon_factor: float = 0.0001
) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """
    Detects external contours and computes convex hulls.
    
    Args:
        image: Input BGR image.
        threshold_val: Binary inverse threshold value.
        blur_kernel: Median filter kernel size.
        approx_epsilon_factor: Epsilon scaling factor for polygon approximation.
        
    Returns:
        Tuple of (grayscale_img, thresh_mask, contours, hierarchy)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, blur_kernel)
    ret, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return gray, thresh, contours, hierarchy


def extract_physical_traits(
    image_path: str,
    output_dir: Optional[str] = None,
    threshold_val: int = 210,
    target_size: Tuple[int, int] = (600, 700)
) -> Dict[str, Any]:
    """
    Extracts physical traits (width, height, area, aspect ratio) from a fruit image
    and renders annotated overlays.
    
    Args:
        image_path: Path to input image.
        output_dir: Directory to save generated debug/visualization images.
        threshold_val: Threshold cutoff.
        target_size: Resized image dimensions (width, height).
        
    Returns:
        Dictionary containing extracted measurements, convex hull data, and paths.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image at path: {image_path}")
    
    img = cv2.resize(img, target_size)
    annotated = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    gray, thresh, cnts, hier = detect_contours_and_hull(img, threshold_val=threshold_val)
    
    # Filter small noise contours
    valid_contours = [c for c in cnts if cv2.contourArea(c) > 500]
    
    measurements = []
    total_w = 0
    total_h = 0
    total_area = 0.0

    for idx, c in enumerate(valid_contours):
        area = float(cv2.contourArea(c))
        total_area += area
        
        # Approximate polygon and compute convex hull
        epsilon = approx_epsilon_factor = 0.0001 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        hull = cv2.convexHull(approx)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0
        
        # Draw contour (reddish) and convex hull (green)
        cv2.drawContours(annotated, [c], -1, (50, 50, 150), 2)
        cv2.drawContours(annotated, [hull], -1, (0, 255, 0), 2)
        
        # Bounding box measurements
        x, y, w, h = cv2.boundingRect(c)
        total_w += w
        total_h += h
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
        
        # Draw measurement dimensions
        cv2.line(annotated, (x, max(0, y - 20)), (x + w, max(0, y - 20)), (0, 255, 0), 2)
        cv2.putText(annotated, f"W: {w}px", (x, max(15, y - 28)), font, 0.7, (0, 255, 0), 2)
        
        cv2.line(annotated, (max(0, x - 20), y), (max(0, x - 20), y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"H: {h}px", (max(5, x - 95), y + h // 2), font, 0.7, (0, 255, 0), 2)
        
        measurements.append({
            "contour_id": idx,
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "aspect_ratio": round(aspect_ratio, 3),
            "area": round(area, 2),
            "hull_area": round(hull_area, 2),
            "solidity": round(solidity, 3)
        })

    # Summary overlay on annotated image
    cv2.putText(annotated, f"Total Width: {total_w} px", (15, 45), font, 0.9, (0, 0, 255), 2)
    cv2.putText(annotated, f"Total Height: {total_h} px", (15, 85), font, 0.9, (0, 0, 255), 2)
    cv2.putText(annotated, f"Contour Count: {len(valid_contours)}", (15, 125), font, 0.8, (255, 100, 0), 2)

    saved_images = {}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out1 = os.path.join(output_dir, "output_image-1.jpg")
        out2 = os.path.join(output_dir, "output_image-2.jpg")
        out3 = os.path.join(output_dir, "output_image-3.jpg")
        
        cv2.imwrite(out1, annotated)
        cv2.imwrite(out2, gray)
        cv2.imwrite(out3, thresh)
        saved_images = {
            "annotated_measurement": out1,
            "grayscale_transform": out2,
            "binary_threshold_mask": out3
        }

    return {
        "status": "success",
        "num_objects": len(valid_contours),
        "total_width_px": total_w,
        "total_height_px": total_h,
        "total_area_px": total_area,
        "measurements": measurements,
        "saved_images": saved_images
    }
