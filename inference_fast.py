"""IMPORT PACKAGES"""
import os
import argparse
import time
import json
import copy
import random
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
import sys

import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator
import matplotlib.colors as clr
from sklearn.metrics import roc_curve, roc_auc_score

from dataset_wle import read_inclusion_combi, augmentations_cls
from train_wle import check_cuda, find_best_model
from model import Model_CLS
import openpyxl  # Add this import for Excel file handling
from itertools import chain
from torch.utils.data import Dataset, DataLoader

""""""""""""""""""""""""
"""" HELPER FUNCTIONS """
""""""""""""""""""""""""


# Make function for defining parameters
def get_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # DEFINE EXPERIMENT NAME
    parser.add_argument('--experimentname', type=str, default=EXPERIMENT_NAME)

    # EXTRACT INFORMATION FROM PARAMETERS USED IN EXPERIMENT
    f = open(os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'params.json'))
    data = json.load(f)
    parser.add_argument('--backbone', type=str, default=data['backbone'])
    parser.add_argument('--imagesize', type=int, default=data['imagesize'])
    parser.add_argument('--num_classes', type=str, default=data['num_classes'])
    parser.add_argument('--label_smoothing', type=float, default=data['label_smoothing'])
    parser.add_argument('--scope', nargs='+', type=str, choices=['Fuji', 'Olympus', 'Pentax'])
    parser.add_argument('--weights', type=str, default='GastroNet')
    parser.add_argument('--names', nargs='+', help='Base experiment name(s) to evaluate')
    parser.add_argument('--data-types', nargs='+', default=['images', 'frames'], choices=['images', 'frames'],
                        help='Data types to evaluate (will be used as a single group)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')

    args = parser.parse_args()

    return args


# Specify function for defining inclusion criteria for training, finetuning and development set
def get_data_inclusion_criteria(scope, data_type=['frames', 'images']):
    criteria = dict()
    criteria['test'] = {'split': ['Validatie set'],
                        'scope': scope, #[s for s in all_scopes if s not in opt.scope],
                        'type': data_type,
                        'min_height': None,
                        'min_width': None
                       }
    #categories = ['Fuji', 'Olympus', 'Pentax']
    
    #criteria = {category: {'scope': [category], 'split': ['Test'], 'min_height': None, 'min_width': None} for category in categories}
    
    return criteria


class CacheDataset(Dataset):
    def __init__(self, inclusion, transform):
        self.inclusion = inclusion
        self.transform = transform

    def __len__(self):
        return len(self.inclusion)

    def __getitem__(self, idx):
        img = self.inclusion[idx]
        file = img['file']
        roi = img['roi']

        image = Image.open(file).convert('RGB')
        image = image.crop((roi[2], roi[0], roi[3], roi[1]))

        image_t = self.transform(image)
        label = float(bool(img.get('label', False)))

        return image_t, label


""""""""""""""""""""""""""""""
"""" FUNCTIONS FOR INFERENCE """
""""""""""""""""""""""""""""""


def run_cls(opt, scope, data_type=['frames', 'images']):
    # Build inclusion list and transforms
    criteria = get_data_inclusion_criteria(scope, data_type)
    val_inclusion = read_inclusion_combi(path=CACHE_PATH, criteria=criteria['test'])
    n_images = len(val_inclusion)
    print(f'Found {n_images} images...')

    data_transforms = augmentations_cls(opt=opt)

    # Dataset + DataLoader for batched loading
    dataset = CacheDataset(val_inclusion, data_transforms['test'])
    dataloader = DataLoader(dataset, batch_size=getattr(opt, 'batch_size', 32),
                            num_workers=getattr(opt, 'num_workers', 4), pin_memory=True)

    # Construct Model and load weights
    model = Model_CLS(opt=opt, inference=True)
    best_index = find_best_model(path=os.path.join(SAVE_DIR, EXPERIMENT_NAME))
    checkpoint = torch.load(os.path.join(SAVE_DIR, EXPERIMENT_NAME, best_index), weights_only=True)['state_dict']

    # Adapt state_dict keys (remove model. prefix)
    checkpoint_keys = list(checkpoint.keys())
    for key in checkpoint_keys:
        checkpoint[key.replace('model.', '')] = checkpoint[key]
        del checkpoint[key]
    model.load_state_dict(checkpoint, strict=False)

    # Push model to GPU and set evaluation mode
    model.cuda()
    model.eval()

    # cuDNN tuning
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    # Batched inference (FP32)
    y_true, y_pred = [], []
    pos = neg = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.cuda(non_blocking=True)
            labels = labels

            # FP32 inference
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).view(-1)

            y_pred.extend(probs.cpu().numpy().tolist())
            y_true.extend([bool(x) for x in labels.tolist()])

            batch_pos = int(sum([1 for v in labels.tolist() if v]))
            batch_neg = labels.size(0) - batch_pos
            pos += batch_pos
            neg += batch_neg

    # Compute AUC for classification
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # sens_val = 0.9
    # index = np.argwhere(np.array(tpr) >= sens_val)[0][0]
    # tpr, fpr, threshold = tpr[index], fpr[index], thresholds[index]

    f1_score = (2 * np.multiply(np.array(tpr), np.array((1-fpr)))) / (np.array(tpr) + np.array((1-fpr)))
    max_f1, index = np.max(f1_score), np.argmax(f1_score)
    tpr, fpr, threshold = tpr[index], fpr[index], thresholds[index]

    print('AUROC {:.4f}'.format(auc))
    # print('\nClassification Performance (>= {}% Sensitivity)'.format(int(sens_val*100)))
    print('sensitivity_cls: {:.4f}'.format(tpr))
    print('specificity_cls: {:.4f}'.format(1-fpr))
    print('threshold: {}'.format(threshold))

    return auc, tpr, fpr, pos, neg, threshold, y_true, y_pred

""""""""""""""""""
"""" EXECUTION """
""""""""""""""""""

def compute_pooled_metrics(all_results):
    """
    all_results: list of dicts, each with keys
        - tpr
        - fpr
        - pos
        - neg
    """
    total_pos = sum(r["pos"] for r in all_results)
    total_neg = sum(r["neg"] for r in all_results)

    pooled_tpr = sum(r["tpr"] * r["pos"] for r in all_results) / total_pos
    pooled_fpr = sum(r["fpr"] * r["neg"] for r in all_results) / total_neg

    pooled_sens = pooled_tpr
    pooled_spec = 1 - pooled_fpr
    return pooled_sens, pooled_spec

def write_header(ws):
    ws.append(["Name", "Scope", "Data type", "AUC", "Mean ± Std",
               "Sensitivity", "Specificity", "Threshold"])


def write_row(ws, name, scope, data_type, aucs, sens_list, spec_list, thr_list):
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ws.append([
        name,
        scope,
        "+".join(data_type),
        ", ".join(f"{r:.3f}" for r in aucs),
        f"{mean_auc:.3f} ± {std_auc:.3f}",
        f"{np.mean(sens_list):.3f}",
        f"{np.mean(spec_list):.3f}",
        ", ".join(f"{thr:.3f}" for thr in thr_list)
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference over cached dataset')
    parser.add_argument('--names', nargs='+', help='Base experiment name(s) to evaluate')
    parser.add_argument('--scopes', nargs='+', default=['Olympus'], help='Scopes to evaluate')
    parser.add_argument('--data-types', nargs='+', default=['images'], choices=['images', 'frames'],
                        help='Data types to evaluate (will be used as a single group)')
    parser.add_argument('--exp-name', default='keyframes', help='Experiment name base for outputs')
    parser.add_argument('--cache-path', default=r'/home/middeljans/COSMO/cache_COSMO_FINAL', help='Path to cache folder')
    parser.add_argument('--output-path', default=r'/home/middeljans/GastroKey/results', help='Output root path')
    parser.add_argument('--save-dir', default=r'/projects/0/prjs1485/GastroKey_experiments/experiments', help='Save dir for experiments')

    args = parser.parse_args()

    names = args.names
    scopes = args.scopes
    data_types = [args.data_types]
    EXP_NAME = args.exp_name
    CACHE_PATH = args.cache_path
    OUTPUT_PATH = args.output_path
    SAVE_DIR = args.save_dir

    output_file = os.path.join(os.path.dirname(OUTPUT_PATH), 'excel_files', f'results_{EXP_NAME}.xlsx')

    for name in names:
        for i in range(5):
            fold_name = name + f'_{i}'
            print(f'evaluating: {fold_name}')
            EXPERIMENT_NAME = fold_name
            opt = get_params()
            for data_type in data_types:
                print(f"\n=== Evaluating {fold_name} on data type: {', '.join(data_type)} ===")
                run_cls(opt, scopes, data_type=data_type)
            print('======')