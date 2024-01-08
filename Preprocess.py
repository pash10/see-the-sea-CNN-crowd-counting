import h5py
import numpy as np
import os
import glob
from scipy.ndimage import gaussian_filter 
import scipy.spatial
from tqdm import tqdm
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(filename='processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gaussian_filter_density(gt, debug=False, visualize=False):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)

    if debug:
        logging.info(f"Ground truth count: {gt_count}")

    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, _ = tree.query(pts, k=4)

    sigmas = []
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0
        sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1 if gt_count > 1 else np.average(np.array(gt.shape)) / 2. / 2.
        sigmas.append(sigma)
        
        if visualize:
            plt.imshow(pt2d)
            plt.title(f"Point {i} before filter")
            plt.show()

        density += gaussian_filter(pt2d, sigma, mode='constant')
        
        if visualize:
            plt.imshow(density)
            plt.title(f"Density map after adding point {i}")
            plt.show()

    if debug:
        logging.info(f"Average sigma: {np.mean(sigmas)}")
        logging.info(f"Sum of density before normalization: {np.sum(density)}")

    # Normalize the density map to make the sum equal to the count of ground truth points
    if gt_count > 0:
        density *= gt_count / np.sum(density)

    if debug:
        logging.info(f"Sum of density after normalization: {np.sum(density)}")

    return density



def process_image(img_path):
    try:
        mat = sio.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
        img = plt.imread(img_path)  # Read image
        k = np.zeros((img.shape[0], img.shape[1]))  # Create a zero matrix of image size
        gt = mat["image_info"][0,0][0,0][0]
        
        # Diagnostic: Print the number of ground truth points
        logging.info(f"Number of GT points for {os.path.basename(img_path)}: {len(gt)}")
        
        # Generate hot encoded matrix of sparse matrix
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
                
        
        # Diagnostic: Assert that the sum of k equals the number of GT points
        assert np.sum(k) == len(gt), "Sum of k does not match number of GT points"
        
        # Generate density map
        k = gaussian_filter_density(k)
        
        # Diagnostic: Check sum of the density map
        density_sum = np.sum(k)
        logging.info(f"Density sum for {os.path.basename(img_path)}: {density_sum}")

        # Save density map in HDF5 format
        file_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
        with h5py.File(file_path, 'w') as hf:
            hf['density'] = k
        logging.info(f"Processed and saved density map for {img_path}")

    except AssertionError as e:
        logging.error(f"Assertion Error for {img_path}: {e}")
    except Exception as e:
        logging.error(f"Error processing image {img_path}: {e}")


def main():
    root = os.path.abspath('ShanghaiTech')
    path_sets = [os.path.join(root, dir) for dir in ['part_A_final/train_data/images', 'part_A_final/test_data/images', 'part_B_final/train_data/images', 'part_B_final/test_data/images']]

    img_paths = [img_path for path_set in path_sets for img_path in glob.glob(os.path.join(path_set, '*.jpg'))]
    for img_path in tqdm(img_paths, desc="Processing images"):
        process_image(img_path)

if __name__ == "__main__":
    main()