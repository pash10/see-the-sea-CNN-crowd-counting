import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
from tensorflow.keras.initializers import RandomNormal
from PIL import Image
import h5py
import os
import cv2
import fnmatch

def gpu():
 # Disable GPU usage by setting environment variable
 os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

 # Verify that GPU is not being used
 if tf.test.gpu_device_name():
    print('GPU found')
 else:
    print('No GPU found, using CPU')


def create_img(path):
    #Function to load,normalize and return image 
    im = Image.open(path).convert('RGB')
    
    im = np.array(im)
    
    im = im/255.0
    
    im[:,:,0]=(im[:,:,0]-0.485)/0.229
    im[:,:,1]=(im[:,:,1]-0.456)/0.224
    im[:,:,2]=(im[:,:,2]-0.406)/0.225

    print(im.shape)
    #im = np.expand_dims(im,axis  = 0)
    return im

def get_input(path):
    print('d')
    img = create_img(path)
    return(img)
    
    
def get_output(path):
    print('c')
    # import target
    # resize target
    
    gt_file = h5py.File(path,'r')
    target = np.asarray(gt_file['density'])
    
    # Resizing the target
    img = cv2.resize(target, (int(target.shape[1]/8), int(target.shape[0]/8)), interpolation=cv2.INTER_CUBIC) * 64
    # Check if target is 2-dimensional, then expand its dimensions
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    print("Resized target shape:", img.shape)  # Add this line
    return img
    

# Image data generator
def image_generator(files, batch_size=1):
    print('a')
    while True:
        if len(files) == 0:
            raise ValueError("No images found in the specified path.")
        input_path = np.random.choice(a=files, size=batch_size)
        
        batch_input = []
        batch_output = []
        
        for path in input_path:
            print('fuck')
            input_img = get_input(path)
            output_target = get_output(path.replace('.jpg', '.h5').replace('images', 'ground_truth'))
            
            batch_input.append(input_img)  # Use append for single items
            batch_output.append(output_target)
        
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        
        yield (batch_x, batch_y)




# Clear any leftovers from previous models or sessions to start fresh
tf.keras.backend.clear_session()

# Define paths for the dataset, assuming the script is run from the root directory of the project
root_dir = os.getcwd()

# Specify the paths to your training and validation datasets
part_A_train = os.path.join(root_dir, 'ShanghaiTech', 'part_B_final', 'train_data', 'images')
part_A_val = os.path.join(root_dir, 'ShanghaiTech', 'part_B_final', 'test_data', 'images')

# Function to get all image paths from a directory
def get_image_paths(directory):
    # This pattern matches both '.jpg' and '.JPG'
    pattern = "*.jpg"
    all_files = os.listdir(directory)
    img_paths = [os.path.join(directory, filename) for filename in all_files if fnmatch.fnmatch(filename.lower(), pattern)]
    return img_paths

# Re-collecting paths with the updated function
train_img_paths = get_image_paths(part_A_train)
val_img_paths = get_image_paths(part_A_val)

print("Total training images:", len(train_img_paths))
print("Total validation images:", len(val_img_paths))

        
# Define a custom callback to save the model at specific intervals of epochs
class ModelCheckpointEveryXEpochs(Callback):
    def __init__(self, save_freq, save_path):
        """
        Initialize the callback.

        Args:
        save_freq (int): Frequency of epochs at which to save the model. For example, `save_freq=5` will save the model every 5 epochs.
        save_path (str): Directory path where the model checkpoints will be saved.

        The constructor saves these arguments in instance variables for later use.
        """
        super(ModelCheckpointEveryXEpochs, self).__init__()  # Call the constructor of the base class (Callback)
        self.save_freq = save_freq  # Number of epochs between saves
        self.save_path = save_path  # Path to save the models

    def on_epoch_end(self, epoch, logs=None):
        """
        This method is called at the end of each epoch. It checks if the model should be saved based on the current epoch number.

        Args:
        epoch (int): The current epoch number, starting from 0.
        logs (dict): A dictionary of metrics results.

        If the current epoch + 1 is divisible by the save frequency, it saves the model to the specified path.
        """
        # Check if the current epoch is one at which we want to save the model
        if (epoch + 5) % self.save_freq == 0:
            # Format the filename with the0 current epoch number
            filename = f"model_epoch_{epoch + 1}.h5"
            # Join the save path with the filename to create the full filepath
            filepath = os.path.join(self.save_path, filename)
            # Save the model to the specified filepath
            self.model.save(filepath)
            # Print a message to inform the user
            print(f"\nSaved model snapshot to {filepath} at epoch {epoch+1}")



def init_weights_vgg(model):
    #vgg =  VGG16(weights='imagenet', include_top=False)
    
    json_file = open('models/VGG_16.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("weights/VGG_16.h5")
    
    vgg = loaded_model
    
    vgg_weights=[]                         
    for layer in vgg.layers:
        if('conv' in layer.name):
            vgg_weights.append(layer.get_weights())
    offset=0
    i=0
    while(i<10):
        if('conv' in model.layers[i+offset].name):
            model.layers[i+offset].set_weights(vgg_weights[i])
            i=i+1
            #print('h')
            
        else:
            offset=offset+1

    return (model)

def save_mod(model , str1 , str2):
    model.save_weights(str1)
    
    model_json = model.to_json()
    
    with open(str2, "w") as json_file:
        json_file.write(model_json)

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


    """
    Define the CrowdNet model architecture.
    
    Parameters:
    input_shape (tuple): The shape of the input images. Default is (None, None, 3) for variable size RGB images.
    batch_norm (bool): Whether to include batch normalization layers.
    kernel (tuple): The size of the convolution kernels.
    init (str): Kernel initializer.
    
    Returns:
    Model: A TensorFlow Keras model compiled with the Adam optimizer and mean squared error loss function.
    """

# Neural network model : VGG + Conv
def CrowdNet(input_shape=(None, None, 3), batch_norm=False):
    """
    Define the CrowdNet model architecture.

    Parameters:
    input_shape (tuple): The shape of the input images. Default is (224, 224, 3) for 224x224 RGB images.

    Returns:
    Model: A TensorFlow Keras model compiled with the Adam optimizer and mean squared error loss function.
    """
    inputs = Input(shape=input_shape)
    x = inputs

    # Initializer
    init = RandomNormal(stddev=0.01)
    # Define the input layer with the specified input shape

    # First layer block: Two convolutional layers with 64 filters, 3x3 kernel size, ReLU activation, and 'same' padding
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_1' ,kernel_initializer=init)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2',kernel_initializer=init)(x)
    # Max pooling layer to reduce the spatial dimensions by half
    x = MaxPooling2D((2, 2), name='max_pooling2d_1')(x)
    
    # Second layer block: Increasing the filters to 128 while keeping other parameters constant
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_3', kernel_initializer=init)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_4',kernel_initializer=init)(x)
    x = MaxPooling2D((2, 2), name='max_pooling2d_2')(x)

    # Third layer block: Further increasing the filters to 256
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_5' ,kernel_initializer=init)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_6' ,kernel_initializer=init )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_7' ,kernel_initializer=init)(x)
    x = MaxPooling2D((2, 2), name='max_pooling2d_3')(x)

    # Fourth layer block: Maxing out the filters at 512 for deeper feature extraction
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_8' ,kernel_initializer=init)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_9' ,kernel_initializer=init)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_10' ,kernel_initializer=init)(x)
    # No max pooling here to preserve spatial dimensions for dense feature maps

    # Repeating the 512 filters block for richer feature extraction
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_11' ,kernel_initializer=init)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_12' ,kernel_initializer=init)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_13' ,kernel_initializer=init)(x)

    # Tapering down the filters towards the output layer
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_14' ,kernel_initializer=init)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_15' ,kernel_initializer=init)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_16' ,kernel_initializer=init)(x)
    # Output layer: A single filter with a 1x1 kernel to predict the density map
    outputs = Conv2D(1, (1, 1), activation='relu', name='conv2d_17' ,kernel_initializer=init)(x)

    # Assembling the model
    model = Model(inputs=inputs, outputs=outputs, name='CrowdNet')
    # Compiling the model with Adam optimizer and mean squared error loss, which is suitable for regression tasks like density estimation
    model.compile(optimizer=Adam(learning_rate=1e-4 * 3), loss=euclidean_distance_loss, metrics=['mae'])

    model = init_weights_vgg(model)
    return model


model = CrowdNet()
model.summary()


gpu() # use cpu or gpu change -1 to use cpu
# Setup for saving model snapshots every X epochs
snapshot_callback = ModelCheckpointEveryXEpochs(save_freq=5, save_path='./model_snapshots')
lr_callback = LearningRateScheduler(lambda epoch, lr: lr * 0.9 if epoch > 10 else lr)
checkpoint_callback = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Assuming you've already defined train_img_paths and val_img_paths
batch_size = 8

#train_steps_per_epoch=10
#val_steps_per_epoch = 10
train_steps_per_epoch = 100
val_steps_per_epoch =100

# Prepare datasets
train_dataset = image_generator(train_img_paths, batch_size)
val_dataset = image_generator(val_img_paths, batch_size)

# Fit model
model.fit(
    train_dataset,
    epochs=5,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=val_steps_per_epoch,
    verbose=1,
    callbacks=[lr_callback, checkpoint_callback, snapshot_callback, early_stopping_callback]
)

save_mod(model,"weights/model_A_weights.h5","models/Model.json")
    