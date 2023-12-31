import h5py

def check_ground_truth_size(file_path):
    with h5py.File(file_path, 'r') as hf:
        # Assuming the ground truth data is stored under the key 'density'
        ground_truth_data = hf['density']
        print("Shape of the ground truth data:", ground_truth_data.shape)

# Replace this with the path to one of your actual ground truth HDF5 files
ground_truth_file = 'ShanghaiTech/part_A_final/test_data/ground_truth/IMG_105.h5'
check_ground_truth_size(ground_truth_file)
