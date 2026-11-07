"""
Support Vector Machine (SVM) Classifier for Apple Quality Grading
Uses image flattening / feature vectors, standard scaling, and linear/RBF kernels.
"""

from typing import Tuple, Optional
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class AppleSVMClassifier:
    """SVM Classifier for fruit feature vector classification."""
    def __init__(self, kernel: str = "linear", C: float = 1.0):
        self.kernel = kernel
        self.C = C
        self.scaler = StandardScaler()
        self.clf = SVC(kernel=kernel, C=C)
        self.is_trained = False

    def load_dataset(
        self, folder_path: str, img_size: Tuple[int, int] = (64, 64)
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Loads and flattens grayscale images from subfolders."""
        images = []
        labels = []
        class_names = sorted(os.listdir(folder_path))

        for class_label, class_name in enumerate(class_names):
            class_path = os.path.join(folder_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                images.append(img.flatten())
                labels.append(class_label)

        return np.array(images), np.array(labels), class_names

    def train_and_evaluate(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
    ) -> dict:
        """Trains SVM and calculates metrics."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.clf.fit(X_train_scaled, y_train)
        self.is_trained = True

        y_pred = self.clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }
