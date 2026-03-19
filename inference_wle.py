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


""""""""""""""""""""""""""""""
"""" FUNCTIONS FOR INFERENCE """
""""""""""""""""""""""""""""""


def run_cls(opt, scope, data_type=['frames', 'images']):
    # Construct data
    criteria = get_data_inclusion_criteria(scope, data_type)
    val_inclusion = read_inclusion_combi(path=CACHE_PATH, criteria=criteria['test'])
    print('Found {} images...'.format(len(val_inclusion)))

    # Construct transforms
    data_transforms = augmentations_cls(opt=opt)

    # Construct Model and load weights
    model = Model_CLS(opt=opt, inference=True)
    best_index = find_best_model(path=os.path.join(SAVE_DIR, EXPERIMENT_NAME))
    checkpoint = torch.load(os.path.join(SAVE_DIR, EXPERIMENT_NAME, best_index), weights_only=True)['state_dict']

    # Adapt state_dict keys (remove model. from the key and save again)
    checkpoint_keys = list(checkpoint.keys())
    for key in checkpoint_keys:
        checkpoint[key.replace('model.', '')] = checkpoint[key]
        del checkpoint[key]
    model.load_state_dict(checkpoint, strict=False)

    # Save final model as .pt file
    #torch.save(model.state_dict(), os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'final_pytorch_model.pt'))
    # weights = torch.load(os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'final_pytorch_model.pt'), weights_only=True)
    # model.load_state_dict(weights, strict=True)

    # Initialize metrics
    pos, neg = 0, 0
    y_true, y_pred = list(), list()

    # Push model to GPU and set in evaluation mode
    model.cuda()
    model.eval()
    with torch.no_grad():

        # Loop over the data
        for img in val_inclusion:

            # Extract information from cache
            file = img['file']
            img_name = os.path.splitext(os.path.split(file)[1])[0]
            roi = img['roi']

            # Construct target
            label = img['label']
            if label:
                target = True
                y_true.append(target)
                pos += 1
            else:
                target = False
                y_true.append(target)
                neg += 1

            # Construct Opening print line
            #print('\nOpening image: {}'.format(img_name))

            # Open Image
            image = Image.open(file).convert('RGB')

            # Crop the image to the ROI
            image = image.crop((roi[2], roi[0], roi[3], roi[1]))

            # Apply transforms to image and mask
            image_t = data_transforms['test'](image)
            image_t = image_t.unsqueeze(0).cuda()

            cls_pred = model(image_t)
            cls_pred = torch.sigmoid(cls_pred).cpu()

            # Append values to list
            y_pred.append(cls_pred.item())

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
    names = ['latent_resnet']

    scopes = ['Olympus', 'Fuji', 'Pentax']
    data_types = [['images']]
    EXP_NAME = 'segmentation_dpt_HL'
    CACHE_PATH = r'/home/middeljans/COSMO/cache_seg' #cache_robust cache_BM cache_seg
    OUTPUT_PATH = r'/home/middeljans/GastroKey/experiments'
    SAVE_DIR = r'/home/middeljans/GastroKey/experiments'

    output_file = f'/home/middeljans/COSMO/excel_files/results_{EXP_NAME}.xlsx'

    for name in names:
        for i in range(5):
            fold_name = name+f'_{i}'
            print('evaluating: {}'.format(fold_name))
            EXPERIMENT_NAME = fold_name
            opt = get_params()
            for data_type in data_types:
                print(f"\n=== Evaluating {fold_name} on data type: {', '.join(data_type)} ===")
                run_cls(opt, scopes, data_type=data_type)
            print('======')