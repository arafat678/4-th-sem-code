import warnings
import matplotlib.pyplot as plt
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
    learn = cnn_learner(dls, resnet34, metrics=[accuracy])

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

    # Path to the image you want to predict
    image_path = "C:\\Users\\Administrator\\Pictures\\apple12.jpg"

    # Load the trained model
    # learn = load_learner("path/to/model.pkl")

    # Load the image
    img = PILImage.create(image_path)

    # Make prediction
    prediction, idx, probabilities = learn.predict(img)
    class_names = learn.dls.vocab
    predicted_class = class_names[idx]

    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted class: {predicted_class} (Probability: {probabilities[idx]:.4f})")
    plt.show()
