# import warnings
# from fastai.vision.all import *

# if __name__ == '__main__':
#     warnings.filterwarnings("ignore", category=UserWarning, module="torch.backends.mps")

#     image_path = "C:\\Users\\Administrator\\Pictures\\apple12.jpg"  # Replace with the actual path to your image

import warnings
from fastai.vision.all import *

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.backends.mps")

    # Replace with your image path
    image_path = "C:\\Users\\Administrator\\Pictures\\apple12.jpg"

    # Replace with your category labels (assume you have 4 categories)
    category_labels = ['apple', 'banana', 'orange', 'grape']

    # Define data blocks for a single image
    img = PILImage.create(image_path)
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=lambda: [img],
        # No validation set for a single image
        splitter=RandomSplitter(valid_pct=0.0),
        get_y=parent_label,
        item_tfms=Resize(460),
        batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]
    )

    # Load pretrained model based on your task (e.g., resnet34 for image classification)
    learn = vision_learner(dblock, resnet34, metrics=accuracy)

    # Fine-tune for a few epochs based on image complexity and dataset size
    learn.fine_tune(5)

    # Predict the category of the image
    pred = learn.predict(img)
    print(f"Predicted category: {pred[0]} ({category_labels[pred[0]]})")
