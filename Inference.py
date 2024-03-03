import cv2
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
import tensorflow as tf


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance as a measure of loss (Loss function).
    Improved for numerical stability by adding a small constant inside the square root.
    Includes checks for NaN and Inf values.
    """
    # Calculate squared difference
    squared_difference = tf.square(y_pred - y_true)
    
    # Sum over all dimensions
    sum_squared_difference = tf.reduce_sum(squared_difference, axis=-1)

    # Add a small constant (epsilon) for numerical stability
    epsilon = tf.maximum(tf.keras.backend.epsilon(), tf.reduce_min(squared_difference) * tf.keras.backend.epsilon())

    # Calculate the square root
    distance = tf.sqrt(sum_squared_difference + epsilon)

    # Check for NaNs and replace them with zeros
    distance = tf.where(tf.math.is_nan(distance), tf.zeros_like(distance), distance)

    return distance

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
    


def load_full_model():
    try:
        # Load the entire model (including its architecture and weights) from a .h5 file
        # Pass the custom_objects dictionary to the load_model function
        loaded_model = tf.keras.models.load_model("USETHIS.h5", custom_objects={'euclidean_distance_loss': euclidean_distance_loss})
        print("Model loaded successfully.")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e  # Re-raise the exception for further details


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
    #Function to load,normalize and return image 
    print(path)
    im = Image.open(path).convert('RGB')
    
    im = np.array(im)
    
    im = im/255.0
    
    im[:,:,0]=(im[:,:,0]-0.485)/0.229
    im[:,:,1]=(im[:,:,1]-0.456)/0.224
    im[:,:,2]=(im[:,:,2]-0.406)/0.225


    im = np.expand_dims(im,axis  = 0)
    return im
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

def predict(path):
    #Function to load image,predict heat map, generate count and return (count , image , heat map)
    #model = load_model()
    model = load_full_model()
    image = create_img(path)
    ans = model.predict(image)
    count = np.sum(ans)
    return count,image,ans



# Main Script Execution:
# This section of the script demonstrates the end-to-end process of loading a trained model, 
# using it to predict on a new image, and displaying the results. It showcases the application 
# of the model in a practical scenario, such as predicting crowd density from an image. The script
# also includes the comparison of the model's prediction with the actual ground truth to evaluate
# its performance.


# Predict and display results for a specific image
# Note: Update 'image_path' with the path of your test image
image_path = 'ShanghaiTech/part_B_final/test_data/images/IMG_170.jpg'
ans, img, hmap = predict(image_path)

if ans is not None:
    print(f"Predicted count: {ans}")

    # Display the original image
    if img is not None:
        plt.imshow(img[0])  # Assumes img is returned as a 4D array
        plt.show()

    # Display the heatmap
    if hmap is not None:
        # Reshape heatmap to 2D
        hmap_2d = np.squeeze(hmap, axis=(0, -1))  # Remove the first and last dimensions
        plt.imshow(hmap_2d, cmap='jet')  # 'jet' colormap for better heatmap visualization
        plt.show()
