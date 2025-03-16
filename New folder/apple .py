import fastai
import os
from fastai.vision.all import *
from fastai.vision import *
import pandas as pd

DATASET_PATH = "C:\\Users\\Administrator\\Desktop\\Quality Grading apple2"
class_names = []

for folder_name in os.listdir(DATASET_PATH):
    class_names.append(folder_name)

print(class_names)
