import os
import json
import cv2
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
import tensorflow as tf
import csv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Assuming your other function definitions (euclidean_distance_loss, load_model, load_full_model, create_img, predict) are here

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


def predict(path):
    #Function to load image,predict heat map, generate count and return (count , image , heat map)
    #model = load_model()
    model = load_full_model()
    image = create_img(path)
    ans = model.predict(image)
    count = np.sum(ans)
    return count,image,ans



def load_ground_truth(h5_path):
    """
    Load ground truth count from an .h5 file.
    Args:
    - h5_path: Path to the .h5 file containing ground truth data.
    Returns:
    - Ground truth count as a float.
    """
    with h5py.File(h5_path, 'r') as hf:
        gt_data = np.array(hf['density'])
        gt_count = np.sum(gt_data)
    return gt_count

def test_all_images(image_dir, gt_dir, output_csv):
    """
    Test all images in a specified directory, compare with ground truth, and write results to a CSV file.
    Args:
    - image_dir: Directory containing images to test.
    - gt_dir: Directory containing .h5 files for ground truth data.
    - output_csv: Path to the output CSV file.
    """
 
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'Ground Truth', 'Predicted Count', 'Difference'])
        
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg'):  # Assuming images are in JPEG format
                image_path = os.path.join(image_dir, filename)
                gt_path = os.path.join(gt_dir, filename.replace('.jpg', '.h5'))
                
                if not os.path.exists(gt_path):
                    print(f"Ground truth file for {filename} does not exist. Skipping.")
                    continue
                
                predicted_count, img, hmap = predict(image_path)
                gt_count = load_ground_truth(gt_path)
                difference = abs(gt_count - predicted_count)
                
                writer.writerow([filename, gt_count, predicted_count, difference])
                print(f"Processed {filename}: GT = {gt_count}, Predicted = {predicted_count}, Diff = {difference}")

# Example usage
image_dir = 'ShanghaiTech/part_B_final/test_data/images'
gt_dir = 'ShanghaiTech/part_B_final/test_data/ground_truth'
output_csv = 'prediction_results.csv'
test_all_images(image_dir, gt_dir, output_csv)