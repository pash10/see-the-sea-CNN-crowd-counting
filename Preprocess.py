import h5py
import scipy.io as io
import numpy as np
import os
import glob
from scipy.ndimage import gaussian_filter 
import scipy.spatial
from tqdm import tqdm
from PIL import Image

# Adjustable Parameters
MAX_SIGMA = 10  # Maximum sigma value for Gaussian filter
MIN_SIGMA = 1.5  # Minimum sigma value for Gaussian filter
TARGET_SIZE = (256, 256)  # Desired image size

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048

    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            sigma = max(min(sigma, MAX_SIGMA), MIN_SIGMA)
        else:
            sigma = np.average([gt.shape[0], gt.shape[1]]) / 2.0

        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    return density

def resize_and_process_image(img_path, target_size):
    try:
        # Resize the image
        with Image.open(img_path) as img:
            img = img.resize(target_size, Image.ANTIALIAS)
            img.save(img_path)

        # Process ground truth
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
        k = np.zeros((target_size[1], target_size[0]), dtype=np.float32)  # Size is (height, width)
        gt = mat["image_info"][0, 0][0, 0][0]
        for x, y in gt:
            if 0 <= int(y) < k.shape[0] and 0 <= int(x) < k.shape[1]:
                k[int(y), int(x)] = 1
        k = gaussian_filter_density(k)

        # Save ground truth
        file_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
        with h5py.File(file_path, 'w') as hf:
            hf['density'] = k

    except Exception as e:
        print(f"Error processing image {img_path}: {e}")

# Main script execution
root = os.path.abspath('ShanghaiTech')
path_sets = [os.path.join(root, 'part_A_final/train_data/images'),
             os.path.join(root, 'part_A_final/test_data/images'),
             os.path.join(root, 'part_B_final/train_data/images'),
             os.path.join(root, 'part_B_final/test_data/images')]

img_paths = [img_path for path in path_sets for img_path in glob.glob(os.path.join(path, '*.jpg'))]

for img_path in tqdm(img_paths, desc="Processing images"):
    resize_and_process_image(img_path, TARGET_SIZE)
