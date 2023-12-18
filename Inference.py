import cv2
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
import tensorflow as tf

# Load and return the neural network model
def load_model():
    try:
        with open('models/Model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights("weights/model_A_weights.h5")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load, normalize and return an image
def create_img(path):
    try:
        im = Image.open(path).convert('RGB')
        im = np.array(im) / 255.0
        im = (im - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        im = np.expand_dims(im, axis=0)
        return im
    except IOError:
        print(f"Error opening image file: {path}")
        return None

# Predict heat map, generate count and return (count, image, heat map)
def predict(model, path):
    image = create_img(path)
    if image is not None and model is not None:
        ans = model.predict(image)
        count = np.sum(ans)
        return count, image, ans
    else:
        return None, None, None

# Main script starts here

# Load the trained model
model = load_model()

# Predict and display results for a specified image
image_path = 'ShanghaiTech/part_A_final/train_data/images/IMG_40.jpg'  # Update with your image path
ans, img, hmap = predict(model, image_path)
if ans is not None:
    # Display predicted count
    print(f"Predicted count: {ans}")
    # Show the original image
    plt.imshow(img.reshape(img.shape[1], img.shape[2], img.shape[3]))
    plt.show()
    # Show the predicted heat map
    plt.imshow(hmap.reshape(hmap.shape[1], hmap.shape[2]), cmap=c.jet)
    plt.show()

# Load and display original count from ground truth
ground_truth_path = 'ShanghaiTech/part_A_final/test_data/ground_truth/IMG_40.h5'  # Update with your ground truth file path
try:
    with h5py.File(ground_truth_path, 'r') as temp:
        temp_1 = np.asarray(temp['density'])
        print(f"Original Count: {int(np.sum(temp_1)) + 1}")
except IOError:
    print(f"Error reading ground truth file: {ground_truth_path}")
