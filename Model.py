import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import h5py
import os
import glob
import cv2
import random

# Clear any previous session
tf.keras.backend.clear_session()

root = os.path.join(os.getcwd(), 'ShanghaiTech')
print(root)

part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
img_paths = [os.path.join(part_A_train, img) for img in os.listdir(part_A_train) if img.endswith('.jpg')]
print("Total images : ", len(img_paths))

def create_img(path):
    im = load_img(path)
    im = img_to_array(im)
    im = im / 255.0
    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
    return im

def get_input(path):
    img = create_img(path)
    return img

def get_output(path):
    with h5py.File(path, 'r') as hf:
        target = np.array(hf['density'])
    target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
    return np.expand_dims(target, axis=-1)

def preprocess_input(image, target):
    crop_size = (int(image.shape[0]/2),int(image.shape[1]/2))
    if random.randint(0,9)<= -1:
        dx = int(random.randint(0,1)*image.shape[0]*1./2)
        dy = int(random.randint(0,1)*image.shape[1]*1./2)
    else:
        dx = int(random.random()*image.shape[0]*1./2)
        dy = int(random.random()*image.shape[1]*1./2)
    img = image[dx : crop_size[0]+dx , dy:crop_size[1]+dy]
    target_aug = target[dx:crop_size[0]+dx,dy:crop_size[1]+dy]
    return(img, target_aug)

def image_generator(files, batch_size=64):
    while True:
        input_path = np.random.choice(a=files, size=batch_size)
        batch_input = []
        batch_output = [] 
        for path in input_path:
            input_img = get_input(path)
            output = get_output(path.replace('.jpg','.h5').replace('images','ground_truth'))
            batch_input.append(input_img)
            batch_output.append(output)
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        yield(batch_x, batch_y)

def euclidean_distance_loss(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1))

def init_weights_vgg(model):
    json_file = open('models/VGG_16.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('weights/VGG_16.h5')
    
    vgg = loaded_model
    vgg_weights = [layer.get_weights() for layer in vgg.layers if 'conv' in layer.name]
    
    offset = 0
    i = 0
    while i < len(vgg_weights):
        if 'conv' in model.layers[i + offset].name:
            model.layers[i + offset].set_weights(vgg_weights[i])
            i += 1
        else:
            offset += 1
    return model

def CrowdNet():  
    rows, cols = None, None
    batch_norm = 0
    kernel = (3, 3)
    init = RandomNormal(stddev=0.01)
    model = Sequential()
    
    if batch_norm:
        model.add(Conv2D(64, kernel_size=kernel, input_shape=(rows,cols,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=kernel, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(128, kernel_size=kernel, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=kernel, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(256, kernel_size=kernel, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, kernel_size=kernel, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, kernel_size=kernel, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(512, kernel_size=kernel, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, kernel_size=kernel, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, kernel_size=kernel, activation='relu', padding='same'))
        model.add(BatchNormalization())
    else:
        model.add(Conv2D(64, kernel_size=kernel, activation='relu', padding='same', input_shape=(rows, cols, 3), kernel_initializer=init))
        model.add(Conv2D(64, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(128, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(Conv2D(128, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(256, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(Conv2D(256, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(Conv2D(256, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(512, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(Conv2D(512, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(Conv2D(512, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))

    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(Conv2D(1, (1, 1), activation='relu', dilation_rate=1, kernel_initializer=init, padding='same'))

    sgd = SGD(lr=1e-7, decay=(5 * 1e-4), momentum=0.95)
    model.compile(optimizer=sgd, loss=euclidean_distance_loss, metrics=['mse'])
    
    return model

model = CrowdNet()
model.compile(optimizer=SGD(lr=1e-7, decay=5e-4, momentum=0.95), 
              loss=euclidean_distance_loss, 
              metrics=['mse'])

# Load image paths and create the training generator
train_gen = image_generator(img_paths, batch_size=1)

# Train the model
model.fit(train_gen, epochs=15, steps_per_epoch=700, verbose=1)

# Save the model weights and architecture
def save_mod(model, weights_path, model_path):
    model.save_weights(weights_path)
    with open(model_path, "w") as json_file:
        json_file.write(model.to_json())

save_mod(model, 'weights/model_A_weights.h5', 'models/Model.json')