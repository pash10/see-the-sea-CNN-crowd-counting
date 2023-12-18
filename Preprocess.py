import h5py
import scipy.io as io
import numpy as np
import os
import glob
from scipy.ndimage import gaussian_filter 
import scipy.spatial
from tqdm import tqdm

def gaussian_filter_density(gt):
    """
    Generate a density map using Gaussian filter transformation.

    Args:
    gt: Ground truth with points marked as 1.

    Returns:
    numpy.ndarray: Density map as numpy array.
    """
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    # Finding the K nearest neighbours using a KDTree
    pts = np.column_stack(np.nonzero(gt))
    tree = scipy.spatial.KDTree(pts, leafsize=2048)
    distances, _ = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros_like(gt, dtype=np.float32)
        pt2d[tuple(pt)] = 1
        sigma = (np.mean(distances[i][1:4]) * 0.1) if gt_count > 1 else (np.average(gt.shape) / 4.)
        density += gaussian_filter(pt2d, sigma, mode='constant')
    
    return density

def process_image(img_path):
    """
    Process each image to create and save a density map.

    Args:
    img_path (str): Path to the image file.
    """
    try:
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
        k = np.zeros((mat["image_info"][0, 0][0, 0][0].shape[0], mat["image_info"][0, 0][0, 0][0].shape[1]), dtype=np.float32)
        gt = mat["image_info"][0, 0][0, 0][0]
        for x, y in gt:
            if 0 <= int(y) < k.shape[0] and 0 <= int(x) < k.shape[1]:
                k[int(y), int(x)] = 1
        k = gaussian_filter_density(k)
        file_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
        with h5py.File(file_path, 'w') as hf:
            hf['density'] = k
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")

# Main script starts here
root = os.path.abspath('ShanghaiTech')
path_sets = [os.path.join(root, 'part_A_final/train_data/images'),
             os.path.join(root, 'part_A_final/test_data/images'),
             os.path.join(root, 'part_B_final/train_data/images'),
             os.path.join(root, 'part_B_final/test_data/images')]

img_paths = [img_path for path in path_sets for img_path in glob.glob(os.path.join(path, '*.jpg'))]

# Processing each image to create and save a density map
for img_path in tqdm(img_paths, desc="Processing images"):
    process_image(img_path)
