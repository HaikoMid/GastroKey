"""IMPORT PACKAGES"""
import random
import os
import json
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision
from torchvision import transforms
from collections import defaultdict

"""DEFINE POSSIBLE EXTENSIONS"""
EXT_VID = ['.mp4', '.m4v', '.avi', '.MP4']
EXT_IMG = ['.jpg', '.png', '.tiff', '.tif', '.bmp']

"""DEFINE VARIABLES OF GASTRONET NORMALIZATION"""
WLE_MEAN = (0.64041256, 0.36125767, 0.31330117)       # WLE RGB VALUES
WLE_STD = (0.18983584, 0.15554344, 0.14093774)        # WLE RGB VALUES


""""""""""""""""""""""""""""""""""""""""""
"""" FUNCTION FOR FINDING INCLUSION """
""""""""""""""""""""""""""""""""""""""""""

def read_inclusion(path, criteria):

    # Initialize lists
    img_list = list()

    # Intialize patient id lists
    patient_neo_list = list()
    patient_ndbe_list = list()

    # Initialize empty dictionary
    cache = dict()

    # Loop over cachefiles and check for inclusion criteria
    cache_files = os.listdir(path)
    for cachefile in cache_files:
        with open(os.path.join(path, cachefile)) as json_file:
            data = json.load(json_file)
            cache[cachefile] = data

    # Obtain inclusion criteria
    split = criteria.get('split', None)
    method = criteria.get('method', None)
    fold = criteria.get('fold', None)
    
    # Loop over keys and values in cache files
    for k_cache, v_cache in cache.items():

        # By default set include to True
        include = True

        # Loop over keys and values in criteria
        for k_ic, v_ic in criteria.items():
            v_val = v_cache[k_ic]
            if isinstance(v_val, list):
                if not any(val in v_ic for val in v_val):
                    include = False
                    break
            else:
                if v_val not in v_ic:
                    include = False
                    break


        if include:
            # Add patients to lists
            if v_cache['class'] == 'neo':
                if v_cache['patient'] not in patient_neo_list:
                    patient_neo_list.append(v_cache['patient'])
            if v_cache['class'] == 'ndbe':
                if v_cache['patient'] not in patient_ndbe_list:
                    patient_ndbe_list.append(v_cache['patient'])

    patient_neo_list.sort()
    np.random.default_rng(seed=11).shuffle(patient_neo_list)

    patient_ndbe_list.sort()
    np.random.default_rng(seed=11).shuffle(patient_ndbe_list)

    # Loop over keys and values in cache files
    for k_cache, v_cache in cache.items():

        # By default set include to True
        include = True

        for k_ic, v_ic in criteria.items():
            v_val = v_cache[k_ic]
            if isinstance(v_val, list):
                if not any(val in v_ic for val in v_val):
                    include = False
                    break
            else:
                if v_val not in v_ic:
                    include = False
                    break
        

        # Check criteria
        if fold is not None:
            if v_cache['kfold'] not in fold:
                include = False

        if v_cache['split'] not in split:
            include = False

        if method is not None:
            if v_cache['method'] not in method:
                include = False

        # Check whether include is true
        if include:
            if v_cache['class'] == 'neo':
                if v_cache['patient'] in patient_neo_list:
                    info = {'file': v_cache['file'], 'label': np.array([1], dtype=np.float32), 'roi': v_cache['roi'], 'scope': v_cache['scope'], 'type': v_cache['type']}
                    img_list.append(info)
            elif v_cache['class'] == 'ndbe':
                if v_cache['patient'] in patient_ndbe_list:
                    info = {'file': v_cache['file'], 'label': np.array([0], dtype=np.float32), 'roi': v_cache['roi'], 'scope': v_cache['scope'], 'type': v_cache['type']}
                    img_list.append(info)
            else:
                print(v_cache['file'], v_cache['class'])
                print('Unrecognized class..')
                raise ValueError
    return img_list

def read_inclusion_combi(path, criteria):

    # Initialize lists
    img_list = list()
    frame_list = list()

    # Intialize patient id lists
    patient_neo_list = list()
    patient_ndbe_list = list()

    # Initialize empty dictionary
    cache = dict()

    # Loop over cachefiles and check for inclusion criteria
    cache_files = os.listdir(path)
    for cachefile in cache_files:
        with open(os.path.join(path, cachefile)) as json_file:
            data = json.load(json_file)
            cache[cachefile] = data

    # Obtain min_height, min_width and mask_only from inclusion criteria
    min_height = criteria.pop('min_height', None)
    min_width = criteria.pop('min_width', None)
    scopes = criteria.get('scope', None)
    image_type = criteria.pop('type', None)
    frame_percentage = criteria.pop('frame_percentage', 100)
    image_percentage = criteria.pop('image_percentage', 100)
    split = criteria.get('split', None)
    exclusive_frames = criteria.pop('exclusive_frames', False)
    seg_frames_only = criteria.pop('seg_frames_only', False)
    fold = criteria.pop('fold', None)
    
    # Loop over keys and values in cache files
    for k_cache, v_cache in cache.items():

        # By default set include to True
        include = True

        # Loop over keys and values in criteria
        for k_ic, v_ic in criteria.items():
            v_val = v_cache[k_ic]
            if isinstance(v_val, list):
                if not any(val in v_ic for val in v_val):
                    include = False
                    break
            else:
                if v_val not in v_ic:
                    include = False
                    break

        if image_type is not None:
            if v_cache['type'] not in image_type:
                include = False

        if include:
            # Add patients to lists
            if v_cache['class'] == 'neo':
                if v_cache['patient'] not in patient_neo_list:
                    patient_neo_list.append(v_cache['patient'])
            if v_cache['class'] == 'ndbe':
                if v_cache['patient'] not in patient_ndbe_list:
                    patient_ndbe_list.append(v_cache['patient'])

    patient_neo_list.sort()
    np.random.default_rng(seed=11).shuffle(patient_neo_list)
    # np.random.default_rng(seed=1).shuffle(patient_neo_list)
    patient_neo_list = patient_neo_list[:int(len(patient_neo_list)*image_percentage/100)]
    #print(patient_neo_list)

    patient_ndbe_list.sort()
    #np.random.default_rng(seed=1).shuffle(patient_ndbe_list)
    np.random.default_rng(seed=11).shuffle(patient_ndbe_list)
    patient_ndbe_list = patient_ndbe_list[:int(len(patient_ndbe_list)*image_percentage/100)]

    #exclusive frames
    image_patients = set()

    # Loop over keys and values in cache files
    for k_cache, v_cache in cache.items():

        # By default set include to True
        include = True

        for k_ic, v_ic in criteria.items():
            v_val = v_cache[k_ic]
            if isinstance(v_val, list):
                if not any(val in v_ic for val in v_val):
                    include = False
                    break
            else:
                if v_val not in v_ic:
                    include = False
                    break
        
        if v_cache['split'] not in split:
            include = False

        # Check whether min_height from inclusion criteria is not None
        if min_height is not None:
            if v_cache['height'] < min_height:
                include = False

        # Check whether min_width from inclusion criteria is not None
        if min_width is not None:
            if v_cache['width'] < min_width:
                include = False

        if scopes is not None:
            if v_cache['scope'] not in scopes:
                include = False

        if fold is not None:
            if v_cache['kfold'] not in fold:
                include = False

        if image_type is not None:
            if v_cache['type'] not in image_type:
                include = False

        if seg_frames_only:
            if v_cache['type'] == 'frames' and len(v_cache['masks']) == 0 and v_cache['class'] == 'neo':
                include = False

        # Check whether include is true
        if include:
            if v_cache['type'] == 'images':
                if v_cache['class'] == 'neo':
                    if v_cache['patient'] in patient_neo_list:
                        info = {'file': v_cache['file'], 'label': np.array([1], dtype=np.float32), 'roi': v_cache['roi'], 'mask': v_cache['masks'], 'scope': v_cache['scope'], 'type': v_cache['type']}
                        img_list.append(info)
                        image_patients.add(v_cache['patient'])
                elif v_cache['class'] == 'ndbe':
                    if v_cache['patient'] in patient_ndbe_list:
                        info = {'file': v_cache['file'], 'label': np.array([0], dtype=np.float32), 'roi': v_cache['roi'], 'mask': v_cache['masks'], 'scope': v_cache['scope'], 'type': v_cache['type']}
                        img_list.append(info)
                        image_patients.add(v_cache['patient'])
                else:
                    print(v_cache['file'], v_cache['class'])
                    print('Unrecognized class..')
                    raise ValueError
            else:
                if v_cache['class'] == 'neo':
                    info = {'file': v_cache['file'], 'label': np.array([1], dtype=np.float32), 'roi': v_cache['roi'], 'mask': v_cache['masks'], 'scope': v_cache['scope'], 'patient': v_cache['patient'], 'type': v_cache['type']}
                    frame_list.append(info)
                elif v_cache['class'] == 'ndbe':
                    info = {'file': v_cache['file'], 'label': np.array([0], dtype=np.float32), 'roi': v_cache['roi'], 'mask': v_cache['masks'], 'scope': v_cache['scope'], 'patient': v_cache['patient'], 'type': v_cache['type']}
                    frame_list.append(info)
                else:
                    print(v_cache['file'], v_cache['class'])
                    print('Unrecognized class..')
                    raise ValueError
        
    if exclusive_frames:
        frame_list = [
            f for f in frame_list
            if f['patient'] not in image_patients
        ]

    if frame_percentage > 0:
        rng = random.Random(42)
        frame_list = rng.sample(frame_list, k=int(len(frame_list) * frame_percentage/100))
    else:
        frame_list = list()

    return img_list + frame_list

def sample_weights(img_list, balance_classes=False, balance_scopes=False):
    # Default: assign 1.0 to all samples (no weighting)
    if not balance_classes and not balance_scopes:
        return np.ones(len(img_list), dtype=np.float64)

    # Count samples per (scope, label) or just (label)
    counts = defaultdict(int)

    for item in img_list:
        scope = item.get('scope') if balance_scopes else 'global'
        label = int(item['label'][0])  # assuming label is np.array([0]) or [1]
        key = (scope, label) if balance_classes else scope
        counts[key] += 1

    # Assign weights
    weights = []
    for item in img_list:
        scope = item.get('scope') if balance_scopes else 'global'
        label = int(item['label'][0])
        key = (scope, label) if balance_classes else scope

        n = counts[key]
        if balance_classes:
            # Weight per class (0.5 for each class per scope or globally)
            class_weight = (1.0 / 2.0) * (1.0 / n) if n > 0 else 0.0
        else:
            # Weight by inverse count per scope or global
            class_weight = 1.0 / n if n > 0 else 0.0

        assert class_weight > 0, f"Zero weight for sample: scope={scope}, label={label}"
        weights.append(class_weight)

    return np.array(weights, dtype=np.float64)

""""""""""""""""""""""""""""""""""""""""""
"""" FUNCTION FOR CREATING TRANSFORMS """
""""""""""""""""""""""""""""""""""""""""""

def augmentations_cls(opt):

    # Initialize lists and dictionary
    train_transforms = list()
    val_transforms = list()
    test_transforms = list()
    data_transforms = dict()

    # Specify augmentation techniques for training
    #train_technique = Identity_CLS()
    train_technique = RandomResizedCrop_CLS(opt.imagesize)

    if opt.backbone == 'DINOv2':
        WLE_MEAN = (0.485, 0.456, 0.406)      # WLE RGB VALUES
        WLE_STD = (0.229, 0.224, 0.225)         # WLE RGB VALUES
    else:
        WLE_MEAN = (0.64041256, 0.36125767, 0.31330117)       # WLE RGB VALUES
        WLE_STD = (0.18983584, 0.15554344, 0.14093774)        # WLE RGB VALUES
        
    train_transforms.extend([train_technique,
                            RandomHorizontalFlip_CLS(p=0.5),
                            RandomVerticalFlip_CLS(p=0.5),
                            Rotate_CLS([0, 90, 180, 270, 360]),
                            ToTensor_CLS(),
                            Normalize_CLS(mean=[WLE_MEAN[0], WLE_MEAN[1], WLE_MEAN[2]],
                                        std=[WLE_STD[0], WLE_STD[1], WLE_STD[2]])])

    # Specify augmentation techniques for validation set
    val_transforms.extend([Resize_CLS([opt.imagesize, opt.imagesize]),
                            ToTensor_CLS(),
                            Normalize_CLS(mean=[WLE_MEAN[0], WLE_MEAN[1], WLE_MEAN[2]],
                                            std=[WLE_STD[0], WLE_STD[1], WLE_STD[2]])])

    # Specify augmentation techniques for test set
    test_transforms.extend([Resize_CLS([opt.imagesize, opt.imagesize]),
                            ToTensor_CLS(),
                            Normalize_CLS(mean=[WLE_MEAN[0], WLE_MEAN[1], WLE_MEAN[2]],
                                          std=[WLE_STD[0], WLE_STD[1], WLE_STD[2]])])

    # Compose transforms and place into dictionary
    data_transforms['train'] = Compose_CLS(train_transforms)
    data_transforms['val'] = Compose_CLS(val_transforms)
    data_transforms['test'] = Compose_CLS(test_transforms)

    return data_transforms

""""""""""""""""""""""""""""""""""""""""""
"""" DATASET FOR TRAINING AND TESTING """
""""""""""""""""""""""""""""""""""""""""""

class DATASET_TRAIN_VAL_TEST_CLS(Dataset):
    def __init__(self, inclusion, transform=None, random_noise=False):
        self.inclusion = inclusion
        self.transform = transform
        self.random_noise = random_noise

    def __len__(self):
        return len(self.inclusion)

    def __getitem__(self, idx):
        img_name = self.inclusion[idx]['file']
        roi = self.inclusion[idx]['roi']
        label = self.inclusion[idx]['label']
        image = Image.open(img_name).convert('RGB')

        # Crop the image to the ROI
        image = image.crop((roi[2], roi[0], roi[3], roi[1]))

        if self.transform:
            image = self.transform(image)

        if self.random_noise:
            ch, row, col = image.shape
            mean = 0
            var = random.choice([0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.05])
            sigma = var ** 0.5
            gauss = torch.tensor(np.random.normal(mean, sigma, (ch, row, col)), dtype=torch.float32)
            image = image + gauss

        return image, label
    

""""""""""""""""""""""""""""""""""""""""""
"""" DATA AUGMENTATION (CLASSIFICATION) """
""""""""""""""""""""""""""""""""""""""""""

# Custom Resize class
class Resize_CLS:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        img = transforms.functional.resize(img, size=self.target_size,
                                           interpolation=transforms.InterpolationMode.LANCZOS)
        return img


# Custom Rotation transform
class Rotate_CLS:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        img = transforms.functional.rotate(img, angle)

        return img


# Custom Horizontal Flipping
class RandomHorizontalFlip_CLS:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        do_horizontal = random.random() < self.p
        if do_horizontal:
            img = transforms.functional.hflip(img)

        return img


# Custom Vertical Flipping
class RandomVerticalFlip_CLS:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        do_vertical = random.random() < self.p
        if do_vertical:
            img = transforms.functional.vflip(img)

        return img


# Custom Class for NO augmentation at all
class Identity_CLS:
    def __init__(self):
        self.identity = None

    def __call__(self, img):

        return img

# Custom standard train augmentation  
class StandardTrain_CLS:
    def __init__(self, opt):
        self.identity = None
        self.opt = opt

    def __call__(self, img):
        transform_list = list()
        train_technique1 = random.choice([
            Identity_CLS(),
            Identity_CLS(),
            GaussianBlur_CLS(),
            RandomAdjustSharpness_CLS(sharpness_factor=2, p=1),
            RandomAffine_CLS(max_rotate=25, max_translate=5, max_shear=15),
        ])
        train_technique2 = random.choice([
            Resize_CLS([self.opt.imagesize, self.opt.imagesize]),
            RandomResizedCrop_CLS((self.opt.imagesize, self.opt.imagesize)),
            RandomResizedCrop_CLS((self.opt.imagesize, self.opt.imagesize), scale=(0.7, 1.1)),
        ])
        train_technique3 = random.choice([
            Identity_CLS(),
            Grayscale_CLS(num_output_channels=3),
            ColorJitter_CLS(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0),
            ColorJitter_CLS(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
            ColorJitter_CLS(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
        ])
        transform_list.extend([train_technique1, train_technique2, train_technique3])
        transform = Compose_CLS(transform_list)
        img = transform(img)
    
        return img

# Custom standard val augmentation    
class StandardVal_CLS:
    def __init__(self, opt):
        self.identity = None
        self.opt = opt

    def __call__(self, img):
        transform_list = list()
        val_technique1 = random.choice([
            Resize_CLS([self.opt.imagesize, self.opt.imagesize]),
            RandomResizedCrop_CLS((self.opt.imagesize, self.opt.imagesize)),
        ])
        val_technique2 = random.choice([
            Identity_CLS(),
            ColorJitter_CLS(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
            ColorJitter_CLS(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
        ])
        transform_list.extend([val_technique1, val_technique2])
        transform = Compose_CLS(transform_list)
        img = transform(img)
    
        return img
    

# Custom Class for Grayscale transform
class Grayscale_CLS:
    def __init__(self, num_output_channels=3):
        self.grayscale = transforms.Grayscale(num_output_channels=num_output_channels)

    def __call__(self, img):
        img = self.grayscale(img)

        return img


# Custom Class for ColorJitter
class ColorJitter_CLS:
    def __init__(self, brightness, contrast, saturation, hue=0.0):
        self.jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                             saturation=saturation, hue=hue)

    def __call__(self, img):
        img = self.jitter(img)

        return img


# Custom Random Adjusting Sharpness
class RandomAdjustSharpness_CLS:
    def __init__(self, sharpness_factor, p=1.):
        self.sharpness = transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=p)

    def __call__(self, img):
        img = self.sharpness(img)

        return img


# Custom Class for Gaussian Blurring images
class GaussianBlur_CLS(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=1.0, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img_blurr = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

        return img_blurr


# Custom Class for making random resized crops for input images, and resize to desired size afterwards
class RandomResizedCrop_CLS(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.9, 1.1), ratio=(7. / 8., 8. / 7.),
                 interpolation=torchvision.transforms.InterpolationMode.LANCZOS):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = img.size
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        img = torchvision.transforms.functional.resized_crop(
            img=img,
            top=i,
            left=j,
            height=h,
            width=w,
            size=self.size,
            interpolation=self.interpolation)

        return img


# Custom Class for random affine transformations with predefined ranges for rotation, translation and shearing
class RandomAffine_CLS:
    def __init__(self, max_rotate, max_translate, max_shear):
        self.max_rotate = max_rotate
        self.max_translate = max_translate
        self.max_shear = max_shear

    def __call__(self, img):
        angle = random.uniform(-self.max_rotate, self.max_rotate)
        translate = [int(random.uniform(-self.max_translate, self.max_translate)),
                     int(random.uniform(-self.max_translate, self.max_translate))]
        shear = [random.uniform(-self.max_shear, self.max_shear)]

        img = transforms.functional.affine(
            img=img,
            angle=angle,
            translate=translate,
            scale=1.,
            shear=shear,
            interpolation=transforms.InterpolationMode.BICUBIC)

        return img


# Custom Class for Normalizing images
class Normalize_CLS:
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img):
        img = self.normalize(img)

        return img


# Custom Class for PIL to Tensor
class ToTensor_CLS:
    def __init__(self):
        self.ToTensor = transforms.ToTensor()

    def __call__(self, img):
        img = self.ToTensor(img)

        return img


# Custom Class for composing
class Compose_CLS:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img