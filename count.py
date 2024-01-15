import cv2
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
import tensorflow as tf


# Function: load_model
# This function is designed to load a pre-trained neural network model from specified files.
# It first reads the model's architecture from a JSON file and then loads the pre-trained weights
# from an H5 file. This approach allows for the separation of model architecture and weights,
# facilitating easier updates and modifications to either component. In machine learning workflows,
# especially in deep learning, this separation is useful for deploying models trained on different
# datasets or for fine-tuning a model on new data without altering its underlying architecture.
#
# The function employs error handling to manage issues that might arise during the loading process,
# such as file not found errors, or problems in model reconstruction from the JSON data. In case of
# any exceptions, it captures and prints the error message, returning `None` to indicate that the
# model loading was unsuccessful.
#
# Args:
# None
#
# Returns:
# tensorflow.keras.models.Model or None: The loaded Keras model if successful, or None if an error occurs.
#
# Usage of this function is essential in scenarios where a model trained and saved in a previous session
# needs to be reloaded for further use, such as additional training, evaluation, or inference.

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

# Function: create_img
# This function handles the loading and preprocessing of an image file for neural network input. 
# The image is first loaded and converted to RGB format. It then undergoes a normalization process 
# where pixel values are scaled to a range of 0 to 1. Following this, standardization is applied 
# where each color channel is adjusted based on predefined mean and standard deviation values. 
# This process is crucial to prepare the image for consistent processing by the neural network, 
# as it aligns the input data format with the format expected by the model during training.
#
# The function includes error handling to catch and report issues in opening the image file. 
# In case of an error, it prints an informative message and returns `None`, indicating the failure 
# to process the image.
#
# Args:
# path (str): Path to the image file to be processed.
#
# Returns:
# numpy.ndarray or None: The preprocessed image as a numpy array if successful, or None if an error occurs.
#
# This preprocessing step is essential in machine learning workflows involving image data, ensuring
# that the model receives input in a format that matches how it was trained, leading to more reliable
# predictions.
def create_img(path):
    try:
        im = Image.open(path).convert('RGB')
        # Resize the image to match the model's expected input size
        im = im.resize((224, 224), Image.ANTIALIAS)
        im = np.array(im) / 255.0
        im = (im - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        im = np.expand_dims(im, axis=0)
        return im
    except IOError:
        print(f"Error opening image file: {path}")
        return None
# Function: predict
# This function is designed to utilize a trained neural network model to predict outputs from an input image.
# It first preprocesses the image using the 'create_img' function. Once the image is in the correct format,
# the model predicts the output, which in the context of crowd counting or similar tasks, is typically a 
# heatmap representing density or object counts. The function then sums up the values in the heatmap to 
# provide a total count, offering a quantitative measure of the objects or features of interest in the image.
#
# The function is robust to situations where the image or model might not be available or correctly loaded, 
# returning None in such cases. This is crucial for error handling in a larger application or workflow.
#
# Args:
# model (tensorflow.keras.models.Model): The trained neural network model used for prediction.
# path (str): Path to the image file to be processed and used for prediction.
#
# Returns:
# tuple: A tuple containing the total count, preprocessed image, and the heatmap (ans). 
#        Returns (None, None, None) if the image or model is not available.
#
# This function is a key component in applications where neural network models are used for image analysis,
# such as estimating crowd sizes or detecting objects in images. It bridges the gap between raw image data
# and actionable insights or quantitative measures derived from those images.

def predict(model, path):
    image = create_img(path)

    # Predict using the model
    ans = model.predict(image)

    # Summing up the values in the heatmap for the count
    # This assumes the model output is a density map where the count is the sum of pixel values
    count = np.sum(ans)

    # Handling different shapes of model output
    if len(ans.shape) == 4:  # If the output shape is (1, height, width, channels)
        # Reshape the heatmap to (height, width), using the first channel
        hmap = ans[0, :, :, 0]
    elif len(ans.shape) == 3:  # If the output is already (height, width, channels)
        # Use the first channel of the heatmap
        hmap = ans[:, :, 0]
    elif len(ans.shape) == 2:  # If the output is (1, features) which might be the case here
        # Directly use the output as the heatmap might not be applicable
        hmap = ans
    else:
        # Handle other cases or raise an error
        print("Unexpected model output shape:", ans.shape)
        return None, None, None

    return count, image, hmap

#fuc for test
def get_num_of_people(model, path):
    """
    Predict the number of people in an image using a trained model.

    Args:
    model (tf.keras.Model): A trained model that outputs a density map.
    path (str): The path to the image file.

    Returns:
    int: The estimated number of people in the image.
    """

    # Call the predict function to get the estimated count
    count, _, _ = predict(model, path)

    # If count is None, the prediction failed, so return None
    if count is None:
        print("Prediction failed for image:", path)
        return None

    # Return the count as an integer
    return int(count)
# Function: load_model
# This function is designed to load a pre-trained neural network model from specified files.
# It first reads the model's architecture from a JSON file and then loads the pre-trained weights
# from an H5 file. This approach allows for the separation of model architecture and weights,
# facilitating easier updates and modifications to either component. In machine learning workflows,
# especially in deep learning, this separation is useful for deploying models trained on different
# datasets or for fine-tuning a model on new data without altering its underlying architecture.
#
# The function employs error handling to manage issues that might arise during the loading process,
# such as file not found errors, or problems in model reconstruction from the JSON data. In case of
# any exceptions, it captures and prints the error message, returning `None` to indicate that the
# model loading was unsuccessful.
#
# Args:
# None
#
# Returns:
# tensorflow.keras.models.Model or None: The loaded Keras model if successful, or None if an error occurs.
#
# Usage of this function is essential in scenarios where a model trained and saved in a previous session
# needs to be reloaded for further use, such as additional training, evaluation, or inference.

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

# Function: create_img
# This function handles the loading and preprocessing of an image file for neural network input. 
# The image is first loaded and converted to RGB format. It then undergoes a normalization process 
# where pixel values are scaled to a range of 0 to 1. Following this, standardization is applied 
# where each color channel is adjusted based on predefined mean and standard deviation values. 
# This process is crucial to prepare the image for consistent processing by the neural network, 
# as it aligns the input data format with the format expected by the model during training.
#
# The function includes error handling to catch and report issues in opening the image file. 
# In case of an error, it prints an informative message and returns `None`, indicating the failure 
# to process the image.
#
# Args:
# path (str): Path to the image file to be processed.
#
# Returns:
# numpy.ndarray or None: The preprocessed image as a numpy array if successful, or None if an error occurs.
#
# This preprocessing step is essential in machine learning workflows involving image data, ensuring
# that the model receives input in a format that matches how it was trained, leading to more reliable
# predictions.
def create_img(path):
    try:
        im = Image.open(path).convert('RGB')
        # Resize the image to match the model's expected input size
        im = im.resize((224, 224), Image.ANTIALIAS)
        im = np.array(im) / 255.0
        im = (im - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        im = np.expand_dims(im, axis=0)
        return im
    except IOError:
        print(f"Error opening image file: {path}")
        return None
# Function: predict
# This function is designed to utilize a trained neural network model to predict outputs from an input image.
# It first preprocesses the image using the 'create_img' function. Once the image is in the correct format,
# the model predicts the output, which in the context of crowd counting or similar tasks, is typically a 
# heatmap representing density or object counts. The function then sums up the values in the heatmap to 
# provide a total count, offering a quantitative measure of the objects or features of interest in the image.
#
# The function is robust to situations where the image or model might not be available or correctly loaded, 
# returning None in such cases. This is crucial for error handling in a larger application or workflow.
#
# Args:
# model (tensorflow.keras.models.Model): The trained neural network model used for prediction.
# path (str): Path to the image file to be processed and used for prediction.
#
# Returns:
# tuple: A tuple containing the total count, preprocessed image, and the heatmap (ans). 
#        Returns (None, None, None) if the image or model is not available.
#
# This function is a key component in applications where neural network models are used for image analysis,
# such as estimating crowd sizes or detecting objects in images. It bridges the gap between raw image data
# and actionable insights or quantitative measures derived from those images.

def fetch_stream():
    global video_stream_url
    video_stream_url = 'http://...'  # Replace with the actual video stream URL from Firebase
    cap = cv2.VideoCapture(video_stream_url)

    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break

def process_frame(model, frame):
    # Convert OpenCV frame to PIL Image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(frame)

    # Now use create_img and predict functions
    processed_frame = create_img(im_pil)
    count, _, _ = predict(model, processed_frame)
    return count

def main():
    model = load_model()
    if model is None:
        print("Model could not be loaded.")
        return

    for frame in fetch_stream():
        count = process_frame(model, frame)
        if count is not None:
            print(f"Count in current frame: {count}")
        # Here, you can add code to update Firebase with the count, if needed

if __name__ == "__main__":
    main()
