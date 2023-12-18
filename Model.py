import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import h5py
import os
import random
import cv2

# Function to preprocess image
# Preprocesses an image by applying normalization and standardization.
# Args:
# path (str): Path to the image file.
# Returns:
# numpy.ndarray: The preprocessed image.

def preprocess_image(path):
    im = load_img(path)
    im = img_to_array(im)
    im /= 255.0
    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
    return im

# Function to get ground truth
# Preprocesses an image by applying normalization and standardization.
# Args:
# path (str): Path to the image file.
# Returns:
# numpy.ndarray: The preprocessed image.

def get_ground_truth(path):
    with h5py.File(path, 'r') as hf:
        target = np.array(hf['density'])
    target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
    return np.expand_dims(target, axis=-1)

# Image generator function
# Generator that yields batches of images and corresponding ground truth for training.
# Args:
# files (list): List of image file paths.
# batch_size (int): Size of the batch to be generated.
# Yields:
# tuple: A batch of images and their corresponding ground truths.

def image_generator(files, batch_size=64):
    while True:
        input_path = np.random.choice(a=files, size=batch_size)
        batch_input, batch_output = [], []
        for path in input_path:
            input_img = preprocess_image(path)
            output = get_ground_truth(path.replace('.jpg','.h5').replace('images','ground_truth'))
            batch_input.append(input_img)
            batch_output.append(output)
        yield np.array(batch_input), np.array(batch_output)

# CrowdNet model definition
# Defines and compiles the CrowdNet model architecture.
# Args:
# rows (int, optional): Height of the input image.
# cols (int, optional): Width of the input image.
# use_batch_norm (bool): Whether to use Batch Normalization.
# optimizer_name (str): Choice of optimizer, 'sgd' or 'adam'.
# learning_rate (float): Initial learning rate for the optimizer.
# include_dense (bool): Whether to include dense layers.
# dropout_rate (float): Dropout rate for dropout layers.
# Returns:
# tensorflow.keras.models.Model: The compiled Keras Model.

def CrowdNet(rows=None, cols=None, use_batch_norm=False, optimizer_name='sgd', learning_rate=1e-7, include_dense=False, dropout_rate=0.0):
    input_layer = Input(shape=(rows, cols, 3))
    x = input_layer

    for filters in [64, 64, 128, 128, 256, 256, 256, 512, 512, 512]:
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.01))(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        if filters in [64, 128, 256]:
            x = MaxPooling2D(strides=2)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    for filters in [512, 512, 512, 256, 128, 64]:
        x = Conv2D(filters, (3, 3), activation='relu', dilation_rate=2, padding='same', kernel_initializer=RandomNormal(stddev=0.01))(x)

    x = Conv2D(1, (1, 1), activation='relu', dilation_rate=1, kernel_initializer=RandomNormal(stddev=0.01), padding='same')(x)

    if include_dense:
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)

    model = Model(inputs=input_layer, outputs=x)

    if optimizer_name == 'sgd':
        opt = SGD(lr=learning_rate, decay=5e-4, momentum=0.95)
    elif optimizer_name == 'adam':
        opt = Adam(lr=learning_rate)

    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])
    return model

# Function to train and save the model
# Trains the model using a given data generator and saves the model's weights and architecture.
# Args:
# model (tensorflow.keras.models.Model): The Keras model to be trained.
# train_gen (generator): Generator that yields training data.
# epochs (int): Number of epochs for training.
# steps_per_epoch (int): Number of steps per epoch.
# weights_path (str): Path where model weights will be saved.
# model_path (str): Path where model architecture will be saved.
# Returns:
# None
def train_and_save_model(model, train_gen, epochs, steps_per_epoch, weights_path, model_path):
    model.fit(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
    model.save_weights(weights_path)
    with open(model_path, "w") as json_file:
        json_file.write(model.to_json())
    print(f"Model and weights saved at {model_path} and {weights_path} respectively.")

# Main script
tf.keras.backend.clear_session()

root_dir = os.path.join(os.getcwd(), 'ShanghaiTech')
part_A_train = os.path.join(root_dir, 'part_A_final/train_data', 'images')
img_paths = [os.path.join(part_A_train, img) for img in os.listdir(part_A_train) if img.endswith('.jpg')]

model = CrowdNet(224, 224, use_batch_norm=True, optimizer_name='adam', include_dense=True, dropout_rate=0.5)
train_gen = image_generator(img_paths, batch_size=1)
train_and_save_model(model, train_gen, epochs=15, steps_per_epoch=700, weights_path='weights/model_A_weights.h5', model_path='models/Model.json')
