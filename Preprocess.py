import h5py
import scipy.io as io
from PIL import Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
from tqdm import tqdm
from matplotlib import cm as CM


# Function to generate a density map using Gaussian filter transformation
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    print(f"Generating density map for {gt_count} points.")
    
    if gt_count == 0:
        return density

    # Find the K nearest neighbours using a KDTree
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
        else:
            sigma = np.average(np.array(gt.shape))/2./2. # case: 1 point
        # Convolve with the gaussian filter
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    
    return density

# List of all image paths
root = os.path.join(os.getcwd(), 'ShanghaiTech')
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_B_train,part_B_test]
img_paths = []
for path in tqdm(path_sets):
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(f"Total images found: {len(img_paths)}")

# Process each image to create and save a density map
for img_path in tqdm(img_paths):
    print(f"Processing image: {img_path}")
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0, 0][0, 0][0]
    for point in gt:
        if int(point[1]) < img.shape[0] and int(point[0]) < img.shape[1]:
            k[int(point[1]), int(point[0])] = 1
    k = gaussian_filter_density(k)
    print(k)
    file_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    with h5py.File(file_path, 'w') as hf:
        hf['density'] = k
    print(f"Density map saved to: {file_path}")

# Display a specific ground truth density map and the corresponding image
# Replace 22 with the index of the image you want to display
if len(img_paths) > 22:
    file_path = img_paths[22].replace('.jpg','.h5').replace('images','ground')
    print(f"Displaying ground truth for: {file_path}")
    with h5py.File(file_path, 'r') as gt_file:
        groundtruth = np.asarray(gt_file['density'])
    plt.imshow(groundtruth, cmap=CM.jet)
    plt.title('Ground Truth Density Map')
    plt.show()
    print("Sum = ", np.sum(groundtruth))

    image_file_path = file_path.replace('.h5', '.jpg').replace('ground_truth', 'images')
    print(f"Displaying image: {image_file_path}")
    img = Image.open(image_file_path)
    plt.imshow(img)
    plt.title('Corresponding Image')
    plt.show()
else:
    print("Not enough images in img_paths to display the 23rd image.")
