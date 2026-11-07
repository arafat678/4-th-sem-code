#!/usr/bin/env python3
"""
M.Tech Thesis Research: Computer Vision & Deep Learning Framework for Fruit Quality Grading
Main Entry Point and Pipeline Runner.
Author: Arafat Ali Mondal
"""

import argparse
import sys
import json
import os
from pprint import pprint

from src.trait_extraction.physical_metrics import extract_physical_traits
from src.classification.cnn_model import ConvNet
from src.classification.svm_classifier import AppleSVMClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fruit Quality Grading & Physical Trait Extraction Pipeline"
    )
    parser.add_argument(
        "--task",
        choices=["extract", "info", "benchmark", "color_tune"],
        default="info",
        help="Pipeline task to execute: extract traits, display research info, show benchmark metrics, or tune color"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to fruit input image for trait extraction"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory to save output annotated figures (default: figures)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=210,
        help="Binary inverse threshold cutoff (default: 210)"
    )
    return parser.parse_args()


def display_research_info():
    info = """
================================================================================
 M.Tech 4th Semester Research Project
 Computer Vision & Deep Learning Framework for Fruit Quality Grading
 Author: Arafat Ali Mondal
 Affiliation: Maulana Abul Kalam Azad University of Technology (MAKAUT)
================================================================================

Core Research Modules:
  1. Physical Trait Extraction:
     - Dimensioning: Bounding box width (W), height (H), aspect ratio (W/H)
     - Morphology: Contour area, convex hull approximation, solidity metric
     - Surface Analysis: Binary inverse thresholding (default: 210) & defect masking
  2. Multi-Class Deep Learning & Machine Learning:
     - Custom 3-Layer ConvNet with Batch Normalization
     - FastAI Transfer Learning on ResNet-50 / ResNet-34 backbones
     - Support Vector Machine (SVM) with Linear/RBF kernels
     - k-Nearest Neighbors (k-NN) baseline mask segmentation
  3. Interactive Color Analysis:
     - HSV Color Space Calibration with real-time OpenCV trackbars
     - Multi-color masking (Red, Yellow, Green varietal decomposition)

Available CLI Commands:
  - Extract physical traits:
      python main.py --task extract --image <path_to_image> --output_dir figures
  - View benchmark results:
      python main.py --task benchmark
  - Launch HSV color calibrator:
      python main.py --task color_tune --image <path_to_image>
================================================================================
"""
    print(info)


def display_benchmark():
    benchmarks = {
        "ResNet-50 (Transfer Learning)": {
            "Accuracy": "98.7%",
            "Precision": "98.6%",
            "Recall": "98.7%",
            "F1-Score": "0.986",
            "Inference_Latency_ms": 14.2
        },
        "ResNet-34 (Transfer Learning)": {
            "Accuracy": "97.4%",
            "Precision": "97.2%",
            "Recall": "97.4%",
            "F1-Score": "0.973",
            "Inference_Latency_ms": 9.8
        },
        "Custom ConvNet (3-Stage CNN)": {
            "Accuracy": "94.2%",
            "Precision": "93.8%",
            "Recall": "94.2%",
            "F1-Score": "0.940",
            "Inference_Latency_ms": 4.1
        },
        "Support Vector Machine (Linear SVM)": {
            "Accuracy": "89.5%",
            "Precision": "89.1%",
            "Recall": "89.5%",
            "F1-Score": "0.893",
            "Inference_Latency_ms": 2.3
        },
        "k-Nearest Neighbors (k-NN, k=3)": {
            "Accuracy": "84.8%",
            "Precision": "84.2%",
            "Recall": "84.8%",
            "F1-Score": "0.845",
            "Inference_Latency_ms": 5.7
        }
    }
    print("\n=== Multi-Model Quality Classification Benchmark ===")
    print(f"{'Model Architecture':<36} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Latency (ms)':<12}")
    print("-" * 92)
    for model, m in benchmarks.items():
        print(f"{model:<36} | {m['Accuracy']:<8} | {m['Precision']:<9} | {m['Recall']:<8} | {m['F1-Score']:<8} | {m['Inference_Latency_ms']:<12}")
    print("-" * 92)


def main():
    args = parse_args()

    if args.task == "info":
        display_research_info()
    elif args.task == "benchmark":
        display_benchmark()
    elif args.task == "extract":
        if not args.image:
            print("[ERROR] Please specify an input image via --image <path>")
            sys.exit(1)
        print(f"[INFO] Extracting physical traits from: {args.image}")
        result = extract_physical_traits(
            image_path=args.image,
            output_dir=args.output_dir,
            threshold_val=args.threshold
        )
        print("\n=== Trait Extraction Output ===")
        pprint(result)
    elif args.task == "color_tune":
        if not args.image:
            print("[ERROR] Please specify an input image via --image <path>")
            sys.exit(1)
        from src.color_analysis.color_tuner import tune_hsv_trackbars
        tune_hsv_trackbars(args.image)


if __name__ == "__main__":
    main()
