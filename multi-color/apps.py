import fastai
import os 
from fastai.vision.all import*
from fastai.vision import*
import pandas as pd
DATASET_PATH="C:\\Users\\Administrator\\Desktop\\Quality Grading apple2"
class_names = []
for folder_name in os.listdir(DATASET_PATH):
    class_names.append(folder_name)
    image_paths = []
labels = []

for i in range(len(class_names)):
    class_name = class_names[i]
    image_names = os.listdir(os.path.join(DATASET_PATH, class_name))

    for image_name in image_names:
        image_path = os.path.join(DATASET_PATH, class_name, image_name)
        image_paths.append(image_path)
        labels.append(i)
        
df = pd.DataFrame({
    'name': image_paths,
    'label': labels
})
df.to_csv(
    path_or_buf=f'{DATASET_PATH}\ labels.csv',
    index=False
)
dls = ImageDataLoaders.from_folder(
    path=DATASET_PATH,
    item_tfms=Resize(224),
    bs=16,
    batch_tfms=[Normalize.from_stats(*imagenet_stats), RandTransform()],
    valid_pct=0.2
)
model=vision_learner(
    dls=dls,
    arch=models.resnet50, 
    metrics=[accuracy, error_rate]
)
model.fine_tune(
     epochs=2
)
model.show_results()