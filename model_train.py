# imports
import numpy as np
import os
import tensorflow as tf
from utils import create_image_generator
import pandas as pd
from sklearn.model_selection import train_test_split
from model import create_model

# Define hyperparameters and other global variables
# ----------------------------------------------------------------
IMG_WIDTH = 768
IMG_HEIGHT = 768
epochs = 1
batch_size = 10
checkpoint_path = "model_checkpoints/my_checkpoint.data-00000-of-00001.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
# ----------------------------------------------------------------

# Let's define the model's layers
df = pd.read_csv("../data/train_ship_segmentations_v2.csv")

# Create a generator
# First split data
train_df, validate_df = train_test_split(df)


# Create generators
train_generator = create_image_generator(batch_size, train_df)
validate_generator = create_image_generator(batch_size, validate_df)
train_steps = np.ceil(float(train_df.shape[0]) / float(batch_size)).astype(int)
validate_steps = np.ceil(float(validate_df.shape[0]) / float(batch_size)).astype(int)

# create a callback to save model checkpoint
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = create_model()

# Fit model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=validate_generator,
    validation_steps=validate_steps,
    epochs=epochs,
    callbacks=[callback])
