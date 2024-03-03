import os

def get_folder_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def main():
    start_path = '/'  # Change this to the root directory you want to start from, e.g., 'C:\\' on Windows.
    for dirpath, dirnames, filenames in os.walk(start_path):
        print(f"Analyzing {dirpath}...")
        folder_size = get_folder_size(dirpath)
        print(f"The total size of {dirpath} is {folder_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
