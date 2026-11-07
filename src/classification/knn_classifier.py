"""
k-Nearest Neighbors (k-NN) Mask Segmentation and Quality Classifier
"""

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


class AppleKNNClassifier:
    """k-NN Classifier for mask pixel evaluation and baseline classification."""
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors)

    def evaluate_mask_agreement(
        self, ground_truth_mask: np.ndarray, detected_mask: np.ndarray
    ) -> dict:
        """Evaluates segmentation alignment between ground truth and detected binary masks."""
        gt_flat = ground_truth_mask.flatten()
        det_flat = detected_mask.flatten()

        X_train, X_test, y_train, y_test = train_test_split(
            gt_flat, det_flat, test_size=0.2, random_state=42
        )
        self.clf.fit(X_train.reshape(-1, 1), y_train)
        y_pred = self.clf.predict(X_test.reshape(-1, 1))

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist()
        }
