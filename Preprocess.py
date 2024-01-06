import cv2
import h5py
import numpy as np
import os
import glob
from scipy.ndimage import gaussian_filter 
import scipy.spatial
from tqdm import tqdm
from PIL import Image
import scipy.io as sio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def gaussian_filter_density(gt, sigma_factor=0.1):
    density = np.zeros_like(gt, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.column_stack(np.nonzero(gt))
    tree = scipy.spatial.KDTree(pts, leafsize=2048)
    distances, _ = tree.query(pts, k=4)

    for pt, distance in zip(pts, distances):
        pt2d = np.zeros_like(gt, dtype=np.float32)
        pt2d[tuple(pt)] = 1
        sigma = np.mean(distance[1:4]) * sigma_factor if gt_count > 1 else sigma_factor
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density

 
#need
def get_image_dimensions_matplotlib(img_path):
    img = mpimg.imread(img_path)
    height, width = img.shape[:2]
    return width, heigh 
    
def process_image(img_path):
    try:
        mat_file = img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_')
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"Mat file not found: {mat_file}")

        #img = mpimg.imread(img_path)
        height, width = get_image_dimensions_matplotlib(img_path)

        mat = sio.loadmat(mat_file)
        gt_points = mat["image_info"][0, 0][0, 0][0]
        
        # Initialize a blank density map
        k = np.zeros((height, width), dtype=np.float32)

        # Populate the density map with ground truth data
        for x, y in gt_points:
            x, y = int(x), int(y)
            if 0 <= y < height and 0 <= x < width:
                k[y, x] = 1

        k = gaussian_filter_density(k)

        # Save the density map
        file_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
        with h5py.File(file_path, 'w') as hf:
            hf['density'] = k

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")



def main():
    root = os.path.abspath('ShanghaiTech')
    path_sets = [os.path.join(root, dir) for dir in ['part_A_final/train_data/images', 'part_A_final/test_data/images', 'part_B_final/train_data/images', 'part_B_final/test_data/images']]

    img_paths = [img_path for path in path_sets for img_path in glob.glob(os.path.join(path, '*.jpg'))]

    for img_path in tqdm(img_paths, desc="Processing images"):
        process_image(img_path)

if __name__ == "__main__":
    main() 
