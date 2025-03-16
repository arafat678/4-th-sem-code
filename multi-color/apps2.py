import warnings
from fastai.vision.all import *

if __name__ == '__main__':
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

    # Create a CNN Learner for training
    learn = vision_learner(dls, resnet34)

    # Fine-tune the model
    learn.fine_tune(0)

    # Get predictions for the test set
    test_dl = dls.test_dl(get_image_files("C:\\Users\\Administrator\\Desktop\\Multi-class Image Classification\\apple-data"))
    preds, _ = learn.get_preds(dl=test_dl)

    # Display the predictions
    print(preds)



