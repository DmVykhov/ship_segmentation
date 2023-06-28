import os
from model import create_model
import pandas as pd
from utils import create_test_generator, rle_encode
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize

# ----------------------------------------------------------------
batch_size = 10
checkpoint_path = "/model_checkpoints/my_checkpoint.data-00000-of-00001.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
IMG_WIDTH = 768
IMG_HEIGHT = 768
image_shape = (768, 768)
# ----------------------------------------------------------------

# Create a basic model instance
model = create_model()
model.load_weights(checkpoint_path, by_name=True)

# Read data
sub_df = pd.read_csv("../data/sample_submission_v2.csv")

# Prediction
test_generator = create_test_generator(batch_size, sub_df)
prediction = model.predict(test_generator)
# Plot results
fig=plt.figure(figsize=(16, 8))
for index, row in sub_df.head(6).iterrows():
    origin_image = imread('../input/test_v2/'+row.ImageId)
    predicted_image = resize(prediction[index], image_shape).reshape(IMG_WIDTH, IMG_HEIGHT) * 255
    plt.subplot(3, 4, 2*index+1)
    plt.imshow(origin_image)
    plt.subplot(3, 4, 2*index+2)
    plt.imshow(predicted_image)
#
for index, row in sub_df.iterrows():
    predict = prediction[index]
    resized_predict =  resize(predict, (IMG_WIDTH, IMG_HEIGHT)) * 255
    mask = resized_predict > 0.5
    sub_df.at[index,'EncodedPixels'] = rle_encode(mask)
sub_df.to_csv("submission.csv", index=False)
#%%
