import os
import sys
import random
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from mrcnn import model as modellib, utils
from mrcnn.config import Config

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Define configuration for the Mask R-CNN model
class NewsTitleConfig(Config):
    # Set the name of the configuration
    NAME = "news_title"
    # Set the number of GPUs to use
    GPU_COUNT = 1
    # Set the number of images per GPU
    IMAGES_PER_GPU = 1
    # Set the number of classes (background + title)
    NUM_CLASSES = 1 + 1
    # Set the size of the images (must be multiple of 2)
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    # Set the steps per epoch and validation steps
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50

    # Override the RPN anchor scales
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # Override the detection minimum confidence
    DETECTION_MIN_CONFIDENCE = 0.9


# Define the dataset class
class NewsTitleDataset(utils.Dataset):

    def load_news_title(self, dataset_dir, subset):
        # Add the classes
        self.add_class("news_title", 1, "title")
        
        # Load the images
        image_dir = os.path.join(dataset_dir, subset)
        for filename in os.listdir(image_dir):
            if not filename.endswith('.jpg'):
                continue
            image_path = os.path.join(image_dir, filename)
            image = io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                "news_title",
                image_id=filename,
                path=image_path,
                width=width, height=height)
            
    def load_mask(self, image_id):
        # Load the mask for the given image ID
        info = self.image_info[image_id]
        mask_path = info['path'].replace('.jpg', '_mask.png')
        mask = io.imread(mask_path)[:,:,0]
        
        # Create the mask and class IDs
        class_ids = np.array([1], dtype=object)
        return mask, class_ids


# Set the paths and parameters for training
dataset_dir = 'VS'
subset = 'train'
model_dir = 'path/to/model'
config = NewsTitleConfig()
model = modellib.MaskRCNN(mode='training', config=config, model_dir=model_dir)

# Load the dataset
dataset_train = NewsTitleDataset()
dataset_train.load_news_title(dataset_dir, subset)
dataset_train.prepare()

# Train the model
model.train(dataset_train, dataset_train,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='all')