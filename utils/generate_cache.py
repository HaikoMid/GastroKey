"""IMPORT PACKAGES"""
import os
import json
import numpy as np
import pandas as pd
from skimage.measure import label
from scipy import ndimage
from PIL import Image
from numba import jit
import random
from collections import Counter
import shutil

"""SPECIFY EXTENSIONS AND DATA ROOTS"""
EXT_VID = ['.mp4', '.m4v',  '.avi']
EXT_IMG = ['.jpg', '.png', '.tiff', '.tif', '.bmp', '.jpeg']


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""DEFINE SET OF FUNCTIONS FOR SELECTING ROI IN RAW ENDOSCOPE IMAGES"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# Define function for minimum pooling the images
@jit(nopython=True)
def min_pooling(img, g=8):

    # Copy Image
    out = img.copy()

    # Determine image shape and compute step size for pooling
    h, w = img.shape
    nh = int(h / g)
    nw = int(w / g)

    # Perform minimum pooling
    for y in range(nh):
        for x in range(nw):
            out[g * y:g * (y + 1), g * x:g * (x + 1)] = np.min(out[g * y:g * (y + 1), g * x:g * (x + 1)])

    return out


# Define function for finding largest connected region in images
def getlargestcc(segmentation):

    # Use built-in label method, to label connected regions of an integer array
    labels = label(segmentation)

    # Assume at least 1 CC
    assert (labels.max() != 0)  # assume at least 1 CC

    # Find the largest of connected regions, return as True and False
    largestcc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return largestcc


# Define function for finding bounding box coordinates for ROI in images
def bbox(img):

    # Find rows and columns where a True Bool is encountered
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)

    # Find the first and last row/column for the bounding box coordinates
    # cmin = left border, cmax = right border, rmin = top border, rmax = bottom border
    # Usage Image.crop((left, top, right, bottom))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


# Define Complete function for finding ROI bounding box coordinates, by combining previous functions
def find_roi(img):

    # Open image as numpy array
    image = np.array(img, dtype=np.uint8)

    # Compute L1 norm of the image
    norm_org = np.linalg.norm(image, axis=-1)

    # Use Gaussian Filter to capture low-frequency information
    img_gauss = ndimage.gaussian_filter(norm_org, sigma=5)

    # Scale pixel values
    img_scaled = ((norm_org - np.min(img_gauss)) / (np.max(img_gauss) - np.min(img_gauss))) * 255

    # Use minimum pooling
    img_norm = min_pooling(img_scaled, g=8)

    # Find largest connected region with threshold image as input
    th = 10
    largestcc = getlargestcc(img_norm >= th)

    # Obtain cropping coordinates
    rmin, rmax, cmin, cmax = bbox(largestcc)

    return rmin, rmax, cmin, cmax


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""DEFINE FUNCTION FOR CREATING CACHE WITH RANDOM/TARGETED VALIDATION/TRAINING (ALL IMAGES)"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def split_into_folds(lst, n):
    """
    Split a list into n approximately equal-sized folds.

    Args:
        lst (list): The input list to be split.
        n (int): The number of folds to create.

    Returns:
        list of lists: A list containing n folds.
    """
    if n <= 0:
        raise ValueError("Number of folds should be greater than 0")

    fold_size = len(lst) // n
    folds = [lst[i:i + fold_size] for i in range(0, len(lst), fold_size)]

    # Handle the case where the list length is not evenly divisible by n
    if len(lst) % n != 0:
        last_fold = folds.pop()
        folds[-1].extend(last_fold)

    return folds

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""DEFINE FUNCTION FOR CREATING CACHE FOR THE DIFFERENT TEST SETS"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Workaround for duplicate JSON filenames
def get_available_json_path(imgname, storing_folder, max_attempts=10):
    base_name = os.path.splitext(imgname)[0]
    for i in range(max_attempts):
        suffix = '' if i == 0 else f'_{i+1}'
        jsonfile = os.path.join(os.getcwd(), storing_folder, f"{base_name}{suffix}.json")
        if not os.path.exists(jsonfile):
            return jsonfile
    raise FileExistsError(f"No available JSON filename after {max_attempts} attempts for base name: {base_name}")

# def generate_cache(root_dir, output_json_path):
#     print('Generating cache...')
#     os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

#     img_files = []
#     # 1. Collect image files
#     for root, dirs, files in os.walk(root_dir):
#         for name in files:
#             if os.path.splitext(name.lower())[1] in EXT_IMG:
#                 img_files.append(os.path.join(root, name))
#             elif name == 'Thumbs.db':
#                 os.remove(os.path.join(root, name))


#     # 3. Extract unique patient IDs based on the video folder name (e.g., isa_252)
#     patient_ids = []
#     for img in img_files:
#         path_parts = img.split(os.sep)
#         # The case folder is -3 (e.g., isa_252_wle_...)
#         case_folder = path_parts[-3]
#         parts = case_folder.split('_')
#         if len(parts) >= 2:
#             patient_id = parts[0] + '_' + parts[1] # e.g., isa_252
#             if patient_id not in patient_ids:
#                 patient_ids.append(patient_id)

#     # 4. K-Fold Logic
#     patient_ids = sorted(patient_ids)
#     random.seed(6)
#     random.shuffle(patient_ids)
#     splits = 5
#     folds = split_into_folds(patient_ids, splits)

#     all_data = []

#     # 5. Process each image
#     for img in img_files:
#         path_parts = img.split(os.sep)
#         case_folder = path_parts[-3]
#         parts = case_folder.split('_')

#         data = {}
#         # Patient ID from the folder name
#         data['patient'] = '_'.join(parts[:2]) 
#         data['file'] = img
#         data['clinic'] = parts[0]
#         data['video_folder'] = case_folder

#         # Updated Path Mapping based on your example
#         data['method'] = path_parts[-2]     # e.g., 'method'
#         data['type']   = path_parts[-4]     # e.g., 'videos'
#         data['class']  = path_parts[-5]     # e.g., 'neo'
#         data['split']  = path_parts[-6]     # e.g., 'Training set'
#         data['scope']  = path_parts[-7]     # e.g., 'Olympus'

#         # Assign Fold
#         for fold in range(splits):
#             if data['patient'] in folds[fold]:
#                 data['kfold'] = fold

#         # Image properties
#         with Image.open(img) as img_obj:
#             data['width'], data['height'] = img_obj.size
#             frame = np.array(img_obj)
        
#         roi = find_roi(frame)
#         data['roi'] = [float(x) for x in roi]

#         all_data.append(data)

#     # 6. Save JSON
#     with open(output_json_path, 'w') as f:
#         json.dump(all_data, f, indent=4)

#     print(f"Saved {len(all_data)} entries to {output_json_path}")

# """"""""""""""""""""""""""
# """EXECUTION OF FUNCTIONS"""
# """"""""""""""""""""""""""
# if __name__ == '__main__':

#     # Define paths to the folders
#     path = '/projects/0/prjs1485/GastroKey'
#     storing_file = '/home/middeljans/GastroKey/cache_test.json'  
#     generate_cache(path, storing_file)

def generate_cache(root_dir, storing_folder):
    # 1. Setup output directory (created in '../cache folders/storing_folder')
    cache_dir = os.path.join(os.getcwd(), '..', 'cache folders', storing_folder)
    print(f'Generating cache in: {cache_dir}')
    os.makedirs(cache_dir, exist_ok=True)

    img_files = []
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files.append(os.path.join(root, name))
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))

    # 3. Create Patient-based folds (identifying patients from folder -3)
    patient_ids = []
    for img in img_files:
        path_parts = img.split(os.sep)
        case_folder = path_parts[-3] # e.g., isa_252_wle_...
        parts = case_folder.split('_')
        if len(parts) >= 2:
            pid = f"{parts[0]}_{parts[1]}"
            if pid not in patient_ids:
                patient_ids.append(pid)

    patient_ids = sorted(patient_ids)
    random.seed(6)
    random.shuffle(patient_ids)
    
    splits = 5
    folds = split_into_folds(patient_ids, splits)

    # 4. Process each image and SAVE INDIVIDUAL JSON
    for img in img_files:
        print(f'Processing: {img}')
        
        path_parts = img.split(os.sep)
        case_folder = path_parts[-3]
        img_name_only = case_folder + '_' + os.path.basename(img)
        img_key = os.path.splitext(img_name_only)[0]

        # Extract data content
        parts = case_folder.split('_')
        data = {
            'patient': '_'.join(parts[:2]),
            'file': img,
            'clinic': parts[0],
            'video_folder': case_folder,
            # Path hierarchy indices
            'method': path_parts[-2],
            'type':   path_parts[-4],
            'class':  path_parts[-5],
            'split':  path_parts[-6],
            'scope':  path_parts[-7]
        }

        # Assign Fold
        for fold_idx in range(splits):
            if data['patient'] in folds[fold_idx]:
                data['kfold'] = fold_idx

        # Image properties & ROI
        with Image.open(img) as img_obj:
            data['width'], data['height'] = img_obj.size
            frame = np.array(img_obj)
        
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # --- KEY CHANGE: STORE INDIVIDUAL JSON ---
        # Generate a unique path for this specific image's JSON
        json_file_path = get_available_json_path(img_name_only, cache_dir)
        
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    print(f"Finished. Individual cache files saved to {cache_dir}")

""""""""""""""""""""""""""
"""EXECUTION OF FUNCTIONS"""
""""""""""""""""""""""""""
if __name__ == '__main__':

    # Define paths to the folders
    path = '/projects/0/prjs1485/GastroKey'
    storing_file = '/home/middeljans/GastroKey/cache'  
    generate_cache(path, storing_file)