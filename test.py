import h5py
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def check_ground_truth_size(file_path):
    with h5py.File(file_path, 'r') as hf:
        # Assuming the ground truth data is stored under the key 'density'
        ground_truth_data = hf['density']
        print("Shape of the ground truth data:", ground_truth_data.shape)

# Replace this with the path to one of your actual ground truth HDF5 files
ground_truth_file = 'ShanghaiTech/part_A_final/test_data/ground_truth/IMG_106.h5'
check_ground_truth_size(ground_truth_file)

def preprocess_image(path):
    im = load_img(path)
    im = img_to_array(im)
    im /= 255.0
    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
    return im


def load_ground_truth(path):
    with h5py.File(path, 'r') as hf:
        density_map = np.array(hf['density'])
    return density_map

def validate_ground_truth(path_to_image, path_to_ground_truth):
    # Load image and ground truth
    image = preprocess_image(path_to_image)
    ground_truth = load_ground_truth(path_to_ground_truth)

    # Check dimensions
    if image.shape[:2] != ground_truth.shape:
        print("Mismatch in image and ground truth dimensions.")
        return False

    # Check counts
    gt_count = np.sum(ground_truth)
    print(f"Ground truth count: {gt_count}")

    # Add more checks if necessary
    return True

validate_ground_truth('ShanghaiTech/part_A_final/train_data/images/IMG_106.jpg', 'ShanghaiTech/part_A_final/test_data/ground_truth/IMG_106.h5')

def direct_ground_truth_check(path):
    with h5py.File(path, 'r') as hf:
        density_map = np.array(hf['density'])
        return np.sum(density_map)

ground_truth_check = direct_ground_truth_check('ShanghaiTech/part_A_final/test_data/ground_truth/IMG_106.h5')
print("Direct ground truth check:", ground_truth_check)



import matplotlib.pyplot as plt

def visualize_density_map(density_map_path):
    with h5py.File(density_map_path, 'r') as hf:
        density_map = np.array(hf['density'])
    plt.imshow(density_map, cmap='jet')
    plt.colorbar()
    plt.show()

visualize_density_map('ShanghaiTech/part_A_final/test_data/ground_truth/IMG_106.h5')

