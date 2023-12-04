import cv2
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
import tensorflow as tf

def load_model():
    # Load and return neural network model
    try:
        json_file = open('Model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights("weights/model_A_weights1.h5")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_img(path):
    # Load, normalize and return image
    try:
        im = Image.open(path).convert('RGB')
        im = np.array(im) / 255.0
        im = (im - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        im = np.expand_dims(im, axis=0)
        return im
    except IOError:
        print(f"Error opening image file {path}")
        return None

def predict(model, path):
    # Predict heat map, generate count and return (count, image, heat map)
    image = create_img(path)
    if image is not None and model is not None:
        ans = model.predict(image)
        count = np.sum(ans)
        return count, image, ans
    else:
        return None, None, None

model = load_model()

# Replace 'ShanghaiTech/part_A/train_data/images/IMG_150.jpg' with your image path
ans, img, hmap = predict(model, 'ShanghaiTech/part_A/train_data/images/IMG_150.jpg')
if ans is not None:
    print(f"Predicted count: {ans}")
    plt.imshow(img.reshape(img.shape[1], img.shape[2], img.shape[3]))
    plt.show()
    plt.imshow(hmap.reshape(hmap.shape[1], hmap.shape[2]), cmap=c.jet)
    plt.show()

    # Replace 'ShanghaiTech/part_A/test_data/ground_truth/IMG_170.h5' with your ground truth file
    try:
        with h5py.File('ShanghaiTech/part_A/test_data/ground_truth/IMG_170.h5', 'r') as temp:
            temp_1 = np.asarray(temp['density'])
            print(f"Original Count: {int(np.sum(temp_1)) + 1}")
    except IOError:
        print("Error reading ground truth file")
