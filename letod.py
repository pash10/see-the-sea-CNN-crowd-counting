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
import datetime
def gpu():
    print("Attempting to disable GPU and use CPU...")
    # Disable GPU usage by setting environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Ensure TensorFlow doesn't see the GPU
    try:
        # This forces TensorFlow to re-evaluate which devices are available for it to use.
        tf.config.experimental.list_physical_devices()
        print("Set environment variable to ensure using CPU only.")
    except Exception as e:
        print(f"An error occurred while setting environment variables: {e}")
    
    # Verify that GPU is not being used
    if tf.test.gpu_device_name():
        print('GPU found, but it should not be used as per configuration.')
    else:
        print('No GPU found, using CPU as intended.')

gpu()


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
    


def pad_images_and_targets(batch_input, batch_output):
    # Determine the maximum dimensions across all images in the batch
    max_image_height = max(img.shape[0] for img in batch_input)
    max_image_width = max(img.shape[1] for img in batch_input)
    
    # Additionally, find the maximum dimensions across all targets in the batch
    max_target_height = max(target.shape[0] for target in batch_output)
    max_target_width = max(target.shape[1] for target in batch_output)

    padded_inputs = []
    padded_targets = []

    for img, target in zip(batch_input, batch_output):
        # Calculate padding sizes for images
        padding_height = max_image_height - img.shape[0]
        padding_width = max_image_width - img.shape[1]
        
        # Apply symmetric padding to the images
        padded_img = np.pad(img, ((0, padding_height), (0, padding_width), (0, 0)), mode='constant', constant_values=0)
        
        # Calculate padding sizes for targets, using max target dimensions
        padding_height = max_target_height - target.shape[0]
        padding_width = max_target_width - target.shape[1]

        # Apply symmetric padding to the targets
        padded_target = np.pad(target, ((0, padding_height), (0, padding_width), (0, 0)), mode='constant', constant_values=0)
        
        padded_inputs.append(padded_img)
        padded_targets.append(padded_target)

    batch_x = np.array(padded_inputs, dtype=np.float32)
    batch_y = np.array(padded_targets, dtype=np.float32)

    return batch_x, batch_y


# Example usage within the image_generator function

def image_generator(files, batch_size=8):
    print("Generator started")
    while True:  # Loop forever so the generator never terminates
        np.random.shuffle(files)  # Shuffle files
        
        batch_input = []
        batch_output = []

        for path in files:
            input_img = get_input(path)
            output_target = get_output(path.replace('.jpg', '.h5').replace('images', 'ground_truth'))

            batch_input.append(input_img)
            batch_output.append(output_target)

            if len(batch_input) == batch_size:
                # Pad the images and targets in the batch
                batch_x, batch_y = pad_images_and_targets(batch_input, batch_output)

                yield batch_x, batch_y  # Yield the padded batch

                # Reset the batch storage
                batch_input = []
                batch_output = []


# Clear any leftovers from previous models or sessions to start fresh
tf.keras.backend.clear_session()

# Define paths for the dataset, assuming the script is run from the root directory of the project
root_dir = os.getcwd()

# Specify the paths to your training and validation datasets
part_A_train = os.path.join(root_dir, 'ShanghaiTech', 'part_A_final', 'train_data', 'images')
part_A_val = os.path.join(root_dir, 'ShanghaiTech', 'part_A_final', 'test_data', 'images')
part_B_train = os.path.join(root_dir, 'ShanghaiTech', 'part_B_final', 'train_data', 'images')
part_B_val = os.path.join(root_dir, 'ShanghaiTech', 'part_B_final', 'test_data', 'images')

shit = os.path.join(part_A_train,part_B_train)
val_all = os.path.join(part_A_val,part_B_val)


# Function to get all image paths from a directory
def get_image_paths(directory):
    # This pattern matches both '.jpg' and '.JPG'
    pattern = "*.jpg"
    all_files = os.listdir(directory)
    img_paths = [os.path.join(directory, filename) for filename in all_files if fnmatch.fnmatch(filename.lower(), pattern)]
    return img_paths

# Re-collecting paths with the updated function
train_img_paths = get_image_paths(part_A_train) + get_image_paths(part_B_train)
val_img_paths = get_image_paths(part_A_val) + get_image_paths(part_B_val)

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
        if (epoch + 1) % self.save_freq == 0:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"model_epoch_{epoch + 1}_{current_time}.h5"
            filepath = os.path.join(self.save_path, filename)
            self.model.save(filepath)
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


class CustomLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1  # Store current epoch number
        self.total_batches = self.params['steps']
        print(f"\nStarting Epoch {self.current_epoch}/{self.params['epochs']}")

    def on_batch_end(self, batch, logs=None):
        current_step = batch + 1
        # Use stored current epoch number
        print(f"Epoch {self.current_epoch} - Step {current_step}/{self.total_batches}", end='\r')

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {self.current_epoch} completed. Loss: {logs.get('loss', 0):.4f}, MAE: {logs.get('mae', 0):.4f}, Val Loss: {logs.get('val_loss', 0):.4f}, Val MAE: {logs.get('val_mae', 0):.4f}")


def save_mod(model):
    """
    Save the model weights, configuration, and full model to files. The filenames include the current date and time.
    This function assumes predefined base paths for saving.
    """
    # Define the base paths for weights, model configuration, and full model
    weights_base_path = "./weights"
    model_base_path = "./models"
    full_model_base_path = "./full_model"

    # Ensure the directories exist
    os.makedirs(os.path.dirname(weights_base_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_base_path), exist_ok=True)
    os.makedirs(os.path.dirname(full_model_base_path), exist_ok=True)

    # Get the current date and time for the filename
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Append the current date and time to the base paths
    weights_path = f"{weights_base_path}_{current_time}.h5"
    model_path = f"{model_base_path}_{current_time}.json"
    full_model_path = f"{full_model_base_path}_{current_time}"
    
    # Save the model weights
    model.save_weights(weights_path)
    print(f"Model weights saved to {weights_path}")
    
    # Save the model configuration to JSON
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    print(f"Model configuration saved to {model_path}")

    # Save the full model
    model.save(full_model_path)
    print(f"Full model saved to {full_model_path}")

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


# Setup for saving model snapshots every X epochs
custom_logging_callback = CustomLoggingCallback()
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



# Prefetch to improve performance plz work my homie 
train_dataset = tf.data.Dataset.from_generator(
    lambda: image_generator(train_img_paths, batch_size),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, None, None, 3], [None, None, None, 1])
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: image_generator(val_img_paths, batch_size),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, None, None, 3], [None, None, None, 1])
).prefetch(tf.data.AUTOTUNE)


# Fit model
model.fit(
    train_dataset,
    epochs=6,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=val_steps_per_epoch,
    verbose=1,
    callbacks=[lr_callback, checkpoint_callback, snapshot_callback, early_stopping_callback, custom_logging_callback]    
)

save_mod(model)



