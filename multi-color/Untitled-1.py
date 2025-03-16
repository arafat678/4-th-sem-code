# %%
import fastai
import os 
from fastai.vision.all import*
from fastai.vision import*
import pandas as pd


# %%
DATASET_PATH="C:\\Users\\Administrator\\Desktop\\Quality Grading apple2"

# %%
class_names = []
for folder_name in os.listdir(DATASET_PATH):
    class_names.append(folder_name)


# %%
class_names 

# %%
image_paths = []
labels = []

for i in range(len(class_names)):
    class_name = class_names[i]
    image_names = os.listdir(os.path.join(DATASET_PATH, class_name))

    for image_name in image_names:
        image_path = os.path.join(DATASET_PATH, class_name, image_name)
        image_paths.append(image_path)
        labels.append(i)


# %%

df = pd.DataFrame({
    'name': image_paths,
    'label': labels
})


# %%
df.head()

# %%
df.tail()

# %%
df.to_csv(
    path_or_buf=f'{DATASET_PATH}\ labels.csv',
    index=False
)

# %%
dls = ImageDataLoaders.from_folder(
    path=DATASET_PATH,
    item_tfms=Resize(224),
    bs=16,
    batch_tfms=[Normalize.from_stats(*imagenet_stats), RandTransform()],
    valid_pct=0.2
)


# %%
dls.show_batch()

# %%
model=vision_learner(
    dls=dls,
    arch=models.resnet50, 
    metrics=[accuracy, error_rate]
)


# %%
model.fine_tune(
     epochs=5
)

# %%
model.show_results()

# %%
#save model and predict

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import os

image_folder = ['Fuji apple', 'McIntosh apple', 'Scarlet Apple', 'Tydemans Early apple']
nimgs = {}

for i in image_folder:
    nimages = len(os.listdir("C:\\Users\\Administrator\\Desktop\\Fast.AI Data\\test\\" + i))
    nimgs[i] = nimages

plt.figure(figsize=(9, 6))
sns.barplot(x=list(nimgs.keys()), y=list(nimgs.values()))
plt.title('Distribution of different classes in Training Dataset')
plt.xlabel('Apple Types')
plt.ylabel('Number of Images')
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import os

image_folder = ['Fuji apple', 'McIntosh apple', 'Scarlet Apple', 'Tydemans Early apple']
nimgs = {}

for folder in image_folder:
    path = "C:\\Users\\Administrator\\Desktop\\Fast.AI Data//test\\" + folder
    nimages = len(os.listdir(path))
    nimgs[folder] = nimages

plt.figure(figsize=(9, 6))
sns.barplot(x=list(nimgs.keys()), y=list(nimgs.values()))
plt.title('Distribution of different classes in Training Dataset')
plt.xlabel('Apple Types')
plt.ylabel('Number of Images')
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import os

image_folder = ['Fuji apple', 'McIntosh apple', 'Scarlet Apple', 'Tydemans Early apple']
nimgs = {}

for folder in image_folder:
    path = "C:\\Users\\Administrator\\Desktop\\Fast.AI Data//validation\\" + folder
    nimages = len(os.listdir(path))
    nimgs[folder] = nimages

plt.figure(figsize=(9, 6))
sns.barplot(x=list(nimgs.keys()), y=list(nimgs.values()))
plt.title('Distribution of different classes in Training Dataset')
plt.xlabel('Apple Types')
plt.ylabel('Number of Images')
plt.show()

# %%
model.export('fruit_classifier1.pk1')

# %%
# Save the model
model.export(f'{DATASET_PATH}/fruit_classifier1.pkl')

# Load the learner
new_model = load_learner(f'{DATASET_PATH}/fruit_classifier1.pkl')




# %%
new_model.predict(
          item= "C:\\Users\\Administrator\\Desktop\\Quality Grading apple2\\Tydeman's Early apple\\17_100.jpg"
)

# %%
from fastai.vision.all import *

image_path = "C:\\Users\\Administrator\\Pictures\\apple2.jpg"
img = PILImage.create(image_path)
show_image(img)

# Assuming `new_model` is already loaded
prediction, idx, probabilities = new_model.predict(img)
class_names = new_model.dls.vocab
predicted_class = class_names[idx]

print(f"Prediction: {predicted_class} ({prediction})")
print(f"Index: {idx}")
print(f"Probabilities: {probabilities}")


# %%
#test
print(model.recorder.values)

# %%
# from fastai.vision.all import *

# # Define your data block
# data = DataBlock(ImageBlock, CategoryBlock),
#     get_items=get_image_files,
#     splitter=RandomSplitter(valid_pct=0.2, seed=42),
#     get_y=parent_label,
#     item_tfms=Resize(460),
#     batch_tfms=[*aug_transforms(size=224),max_warp=0) ,Normalize.from_stats(*imagenet_stats)]

# # Create a dataloaders object
# dls = dblock.dataloaders("C:\\Users\\Administrator\\Desktop\\Fast.AI Data", bs=32)

# # Create a learner
# learn = cnn_learner(dls, resnet34, metrics=[error_rate, accuracy])

# learn.fine_tune(3)
# interp=ClassificationInterpretation.from_learn(learn)
from fastai.vision.all import *

# Define your data block
data = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(460),
    batch_tfms=[*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)]
)

# Create a dataloaders object
dls = data.dataloaders("C:\\Users\\Administrator\\Desktop\\Fast.AI Data", bs=32)

# Create a learner
learn = cnn_learner(dls, resnet34, metrics=[error_rate, accuracy])

# Fine-tune the model
learn.fine_tune(3)

# Interpret the results
interp = ClassificationInterpretation.from_learner(learn)



# %%


# %%
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Assuming conf_mat is your confusion matrix and classes is the list of class names
# Replace this with your actual confusion matrix and class names

# Example confusion matrix
conf_mat = np.array([[10, 2, 3],
                    [1, 15, 2],
                    [0, 1, 20]])

# Example class names
classes = ['Fuji apple 0', 'McIntosh apple 1', 'Scarlet Apple2','Tydemans Early apple 3']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(2)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot confusion matrix
np.set_printoptions(precision=2)
fig1 = plt.figure(figsize=(7, 6))
plot_confusion_matrix(conf_mat, classes=classes, title='Confusion Matrix')
plt.show()


# %%


# %%

import warnings
from fastai.vision.all import *

# Suppress the deprecated 'has_mps' warning
warnings.filterwarnings("ignore", category=UserWarning, module="torch.backends.mps")

# Define your data block
data = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(460),
    batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]
)

# Create a dataloaders object
dls = data.dataloaders("C:\\Users\\Administrator\\Desktop\\Multi-class Image Classification\\apple-data", bs=32)

# Create a CNN Learner
learn = vision_learner(dls, resnet34, metrics=[accuracy])

# Fine-tune the model
learn.fine_tune(5)

# Plot training and validation loss
learn.recorder.plot_loss()

# Display confusion matrix
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(8, 8))

# Display top losses
interp.plot_top_losses(9, figsize=(15, 10))

# Show results
plt.show()


# %%


# %%


# %%


# %%



