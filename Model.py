import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten, Reshape  
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D  
import tensorflow.keras.backend as K 
import numpy as np
import h5py
import os
import random
from tensorflow.keras.models import model_from_json
import cv2


# Function: preprocess_image
# This function preprocesses an image by applying normalization and standardization, 
# making it suitable for input into a neural network. Normalization scales pixel values 
# to a range of 0 to 1, and standardization adjusts the channels of the image based on 
# predefined mean and standard deviation values. This preprocessing is crucial for 
# consistent model input and improved model performance.
#
# Args:
# path (str): Path to the image file.
#
# Returns:
# numpy.ndarray: The preprocessed image as a numpy array.
#
# The function loads the image from the given path, converts it to an array, 
# normalizes its pixel values, and standardizes each color channel. The result 
# is an image ready for model training or predictions.


def preprocess_image(path):
    im = load_img(path)
    im = img_to_array(im)
    im /= 255.0
    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
    return im

# Function: get_ground_truth
# This function retrieves the ground truth data for an image. In machine learning and computer vision,
# 'ground truth' refers to the accurate, real-world information used as a benchmark to train and evaluate
# models. In tasks like crowd counting, ground truth data often comprises detailed annotations or density
# maps that accurately represent the number of objects or people in an image. These annotations are usually
# created manually by human labelers, ensuring that they reflect the true scenario depicted in the image.
# The ground truth serves as the standard or 'truth' against which a model's predictions are compared, 
# allowing us to assess the model's accuracy and performance.
#
# In this function, the ground truth is represented by a density map stored in an HDF5 file. The density
# map provides a pixel-wise representation of object counts in the image, allowing for fine-grained
# comparison and training. The function reads this map, resizes it for compatibility with the model's input
# dimensions, and adjusts its scale. This processed density map is then used during model training and
# evaluation to compare the predicted density against the actual density.
#
# Args:
# path (str): Path to the ground truth file, typically an HDF5 file containing density maps.
#
# Returns:
# numpy.ndarray: The resized and processed ground truth density map as a numpy array.

def get_ground_truth(path):
    try:
        with h5py.File(path, 'r') as hf:
            target = np.array(hf['density'])

        # Ensure dimensions are divisible by 8
        height, width = target.shape
        if height % 8 != 0 or width % 8 != 0:
            # Compute new dimensions that are divisible by 8
            new_height = height + (8 - height % 8)
            new_width = width + (8 - width % 8)

            # Resize target to new dimensions
            target = cv2.resize(target, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Resize target for model input
        target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
        return np.expand_dims(target, axis=-1)
    except IOError:
        print(f"Ground truth file not found: {path}")
        return None

# Image Generator:
# This function acts as a generator for batch processing in model training. It randomly selects
# images from the provided list, preprocesses them, and pairs them with their corresponding ground truth.
# The generator yields batches of images and ground truth, suitable for training neural network models.
# Args:
# files (list): List of image file paths.
# batch_size (int): Number of images to include in each batch.
# Yields:
# (numpy.ndarray, numpy.ndarray): A tuple containing a batch of images and their corresponding ground truths.
def image_generator(files, batch_size=64, target_height=224, target_width=224):
    while True:
        input_path = np.random.choice(a=files, size=batch_size)
        batch_input, batch_output = [], []
        for path in input_path:
            input_img = preprocess_image(path)

            ground_truth_path = path.replace('.jpg','.h5').replace('images','ground_truth')
            output = get_ground_truth(ground_truth_path)
            
            if output is None:  # Skip this image if ground truth is missing
                continue

            # Ensure the output is resized correctly
            current_height, current_width = output.shape[:2]
            if current_height != target_height or current_width != target_width:
                output = cv2.resize(output, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

            output = np.expand_dims(output, axis=-1)  # Add channel dimension
            batch_input.append(input_img)
            batch_output.append(output)

            if len(batch_input) == batch_size:
                break

        # Yield only if the batch is full, otherwise skip to next iteration to fill the batch
        if len(batch_input) == batch_size:
            yield np.array(batch_input), np.array(batch_output)





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




# Function to initialize the weights of a model with weights from a pre-trained VGG16 model.

    # This function aids in transfer learning by using the powerful feature extraction 
    # capabilities of the VGG16 model, pre-trained on ImageNet. It loads the VGG16 model's 
    # architecture and weights from JSON and H5 files, respectively. The function then 
    # transfers the weights from the convolutional layers of VGG16 to the corresponding layers 
    # of the provided model. This initialization can enhance the learning capability of the model,
    # particularly useful when dealing with limited training data or seeking quick convergence.

    # Args:
    # model (tensorflow.keras.models.Model): The custom model to initialize with VGG16 weights.
    # Returns:
    # tensorflow.keras.models.Model: The model with VGG16 weights initialized.
    
    # Note: The JSON and H5 files must contain the VGG16 model architecture and weights. 
    # Ensure compatibility of convolutional layers between the custom and VGG16 models.


def init_weights_vgg(model):
    # Load VGG16 model architecture and weights
    json_file = open('models/VGG_16.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_vgg_model = model_from_json(loaded_model_json)
    loaded_vgg_model.load_weights("weights/VGG_16.h5")
    
    # Iterate over the layers of VGG16 and your model simultaneously
    vgg_layers = [layer for layer in loaded_vgg_model.layers if 'conv' in layer.name]
    model_layers = [layer for layer in model.layers if 'conv' in layer.name]

    for vgg_layer, model_layer in zip(vgg_layers, model_layers):
        # Check if the layer shapes are compatible
        if vgg_layer.get_weights()[0].shape == model_layer.get_weights()[0].shape:
            model_layer.set_weights(vgg_layer.get_weights())
        else:
            print("fuck")
            break  # Stop if layers are not compatible

    return model


# CrowdNet Model Definition:
# This function defines and compiles the CrowdNet model architecture. It is designed to be flexible 
# with various parameters allowing customization of the input size, batch normalization, choice of optimizer, 
# and inclusion of dropout and dense layers.
#
# Args:
# rows (int, optional): Height of the input image.
# cols (int, optional): Width of the input image.
# use_batch_norm (bool): Flag to use Batch Normalization.
# optimizer_name (str): Choice of optimizer ('sgd' or 'adam').
# learning_rate (float): Initial learning rate for the optimizer.
# include_dense (bool): Flag to include dense layers at the end of the model.
# dropout_rate (float): Dropout rate for dropout layers.
#
# Returns:
# tensorflow.keras.models.Model: A compiled Keras Model based on the specified architecture.
# 
# The model structure includes convolutional layers with options for batch normalization, max pooling,
# and dropout. The model can be further customized and tested for various image processing tasks.
# After defining the model, it is recommended to check the model structure using model.summary().


def CrowdNet(use_batch_norm, optimizer_name='adam', learning_rate=1e-4, dropout_rate=0.5):
    # Input layer
    input_layer = Input(shape=(None, None, 3))
    x = input_layer

    # First set of layers: Conv, BatchNorm, MaxPooling, Dropout
    for filters in [64, 64, 128, 128, 256, 256, 256, 512, 512, 512]:
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.01))(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Dropout(dropout_rate)(x)

    # Additional Conv layers without pooling
    for filters in [512, 512, 512, 256, 128, 64]:
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.01))(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    # Output layer
    x = Conv2D(1, (1, 1), activation='relu', dilation_rate=1, kernel_initializer=RandomNormal(stddev=0.01), padding='same')(x)

    # Creating model
    model = Model(inputs=input_layer, outputs=x)

    # Choose the optimizer
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate, decay=5e-4, momentum=0.95)

    # Compile the model
    model.compile(optimizer=optimizer, loss=euclidean_distance_loss, metrics=['mse'])
    model = init_weights_vgg(model)
    return model
# Function: train_and_save_model
# This function handles the training of a Keras model using provided data from a generator. It 
# allows specifying the number of training epochs and steps per epoch, which are essential parameters 
# in the training process. After training, the model's weights and architecture are saved to the 
# specified paths. This function is particularly useful for training deep learning models in 
# scenarios where data is fed in batches (e.g., large datasets or real-time data processing).
#
# Args:
# model (tensorflow.keras.models.Model): The Keras model to be trained.
# train_gen (generator): Generator that yields training data, typically a batch of inputs and 
#                        corresponding target outputs.
# epochs (int): The total number of iterations over the entire dataset for training the model.
# steps_per_epoch (int): The number of batch iterations before a training epoch is considered finished.
# weights_path (str): The file path to save the trained model weights after training.
# model_path (str): The file path to save the model's architecture in JSON format.
#
# Returns:
# None: This function does not return any value but saves the trained model to specified paths.
#
# The function also prints a message upon successful saving of the model and weights. It is crucial
# to provide valid paths for saving the model and weights to ensure no data loss after training.

def train_and_save_model(model, train_gen, epochs, steps_per_epoch, weights_path, model_path):
    print(model.summary())
    model.fit(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
    model.save_weights(weights_path)
    with open(model_path, "w") as json_file:
        json_file.write(model.to_json())
    print(f"Model and weights saved at {model_path} and {weights_path} respectively.")

# Main Script Execution:
# The script begins by clearing any existing Keras backend session to ensure a fresh environment.
# It then sets up the directory paths for training data and loads the image paths from the specified
# directory. After defining the model using the CrowdNet function with specified parameters, it
# creates a data generator for the training images. The model is then trained using this generator
# with specified epochs and steps per epoch, and the trained model is saved to the given paths for
# weights and architecture.

# Clear any previous Keras sessions to ensure a clean environment
tf.keras.backend.clear_session()

# Define the root directory and path for training images
root_dir = os.path.join(os.getcwd(), 'ShanghaiTech')
part_A_train = os.path.join(root_dir, 'part_A_final/train_data', 'images')
img_paths = [os.path.join(part_A_train, img) for img in os.listdir(part_A_train) if img.endswith('.jpg')]

# ...

# Initialize and compile the CrowdNet model with specified parameters
model = CrowdNet(use_batch_norm=False, optimizer_name='adam', dropout_rate=0.5)

# Create a generator for the training data
train_gen = image_generator(img_paths, batch_size=1)

# Train the model and save it
train_and_save_model(model, train_gen, epochs=15, steps_per_epoch=700, weights_path='weights/model_A_weights.h5', model_path='models/Model.json')
