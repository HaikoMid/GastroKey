"""IMPORT PACKAGES"""
import os
import re
import argparse
import json
from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchmetrics
import pytorch_lightning as pl
import shutil
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from dataset_wle import read_inclusion, sample_weights, augmentations_cls
from dataset_wle import DATASET_TRAIN_VAL_TEST_CLS
from loss_optim_wle import construct_optimizer, construct_scheduler
from loss_optim_wle import construct_loss_function_cls
from model import Model_CLS
import torch.nn.functional as F
import wandb
import random

""""""""""""""""""""""""
"""" HELPER FUNCTIONS """
""""""""""""""""""""""""


# Make function for defining parameters
def get_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # DEFINE EXPERIMENT NAME
    parser.add_argument('--experimentname', type=str, default=EXPERIMENT_NAME)
    parser.add_argument('--basename', type=str, default='test')
    parser.add_argument('--seed', type=int, default=9)

    # DEFINE MODEL
    parser.add_argument('--backbone', type=str, default='DINOv3')
    parser.add_argument('--weights', type=str, default='GastroDINO')
    parser.add_argument('--train_lr', type=float, default=1e-5)

    # DEFINE OPTIMIZER, CRITERION, SCHEDULER
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='Plateau')
    parser.add_argument('--cls_criterion', type=str, default='BCE')
    parser.add_argument('--cls_criterion_weight', type=int, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.01)

    # TRAINING PARAMETERS
    parser.add_argument('--imagesize', type=int, default=256)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)

    # WEIGHT SAMPLING
    parser.add_argument('--weight_sampling', type=str, choices=['class'], default=None)

    # CROSS VALIDATION
    parser.add_argument('--method', nargs='+', type=str, default=['uniform', 'latent'])
    parser.add_argument('--val_fold', nargs='+', type=int, default=[0])

    # TASK
    parser.add_argument('--task', type=str, default='cls')

    args = parser.parse_args()

    return args


# Specify function for defining inclusion criteria for training, finetuning and development set
def get_data_inclusion_criteria(opt):
    criteria = dict()
    train_folds = [0,1,2,3,4]
    for val_fold in opt.val_fold:
        if val_fold in train_folds:
            train_folds.remove(val_fold)

    criteria['train'] = {'split': ['Training set'],
                         'fold': train_folds,
                         'method': opt.method
                            
                        }

    criteria['val'] = {'split': ['Training set'],
                        'fold': opt.val_fold,
                        'method': opt.method
                       }
    
    criteria['test'] = {'split': ['Validatie set'],
                        #'fold': train_folds,
                       }

    return criteria

# Function for checking whether GPU or CPU is being used
def check_cuda():

    print('\nExtract Device...')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(device)
        device_count = torch.cuda.device_count()
        torch.cuda.empty_cache()
        print('Using device: {}'.format(device))
        print('Device name: {}'.format(device_name))
        print('Device count: {}\n'.format(device_count))
    else:
        device = torch.device('cpu')
        print('Using device: cpu\n')


# Find best checkpoint model
def find_best_model(path):

    # Append all files
    files = list()
    values = list()

    # List files with certain extension
    for file in os.listdir(path):
        if file.endswith('.ckpt'):
            val = re.findall(r'\d+\.\d+', file)
            value = val[0]
            files.append(file)
            values.append(value)

    # Find file with highest value
    max_val = max(values)
    indices = [i for i, x in enumerate(values) if x == max_val]
    max_index = indices[-1]

    return files[max_index]


""""""""""""""""""""""""""""""""""""""""""""
"""" DATA: PYTORCH LIGHTNING DATAMODULES """
""""""""""""""""""""""""""""""""""""""""""""
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#why-do-i-need-a-datamodule


class WLEDataModule_CLS(pl.LightningDataModule):
    def __init__(self, data_dir, criteria, transforms, opt):
        super().__init__()
        self.data_dir = data_dir
        self.criteria = criteria
        self.transforms = transforms
        self.train_sampler = None
        self.train_set = None
        self.val_set_train = None
        self.val_set_test = None

    def setup(self, stage: Optional[str] = None):

        # Find data that satisfies the inclusion criteria
        train_inclusion = read_inclusion(path=self.data_dir, criteria=self.criteria['train'])
        val_inclusion = read_inclusion(path=self.data_dir, criteria=self.criteria['val'])
        test_inclusion = read_inclusion(path=self.data_dir, criteria=self.criteria['test'])

        print('train inclusion: {}'.format(len(train_inclusion)))
        print('val inclusion: {}'.format(len(val_inclusion)))
        print('test inclusion: {}'.format(len(test_inclusion)))

        # Construct weights for the samples
        if opt.weight_sampling == 'class':
            train_weights = sample_weights(train_inclusion, balance_classes=True)
            self.train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_inclusion), replacement=True)

        # Construct datasets
        self.train_set = DATASET_TRAIN_VAL_TEST_CLS(inclusion=train_inclusion,
                                                transform=self.transforms['train'],
                                                random_noise=False)

        self.val_set = DATASET_TRAIN_VAL_TEST_CLS(inclusion=val_inclusion,
                                                  transform=self.transforms['val'],
                                                  random_noise=False)
        
        self.test_set = DATASET_TRAIN_VAL_TEST_CLS(inclusion=test_inclusion,
                                                  transform=self.transforms['test'],
                                                  random_noise=False)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=opt.batchsize, shuffle=False, num_workers=4,
                          pin_memory=True, prefetch_factor=4, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=opt.batchsize, num_workers=4,
                          pin_memory=True, prefetch_factor=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=opt.batchsize, num_workers=4)
    

""""""""""""""""""""""""""""""""""""""""""""""""""
"""" MODEL: PYTORCH LIGHTNING & PYTORCH MODULE """
""""""""""""""""""""""""""""""""""""""""""""""""""
# https://www.pytorchlightning.ai/
# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
# https://medium.com/aimstack/how-to-tune-hyper-params-with-fixed-seeds-using-pytorch-lightning-and-aim-c61c73f75c7c
# https://pytorch-lightning.readthedocs.io/en/1.4.3/common/weights_loading.html
# https://pytorch-lightning.readthedocs.io/en/stable/common/production_inference.html


class WLEModel_CLS(pl.LightningModule):
    def __init__(self, opt):
        super(WLEModel_CLS, self).__init__()

        # Fix seed for reproducibility
        pl.seed_everything(seed=opt.seed, workers=True)

        # Define label smoothing
        self.label_smoothing = opt.label_smoothing

        # Define sigmoid activation
        self.sigmoid = nn.Sigmoid()

        # Define loss functions for classification and segmentation
        self.cls_criterion = construct_loss_function_cls(opt=opt)

        # Define model
        self.model = Model_CLS(opt=opt, inference=False)

        # Define trainable parts of the model
        if opt.finetune == True:
            for name, param in self.model.named_parameters():
                param.requires_grad = ("head" in name)

        # Specify metrics
        self.train_auc = torchmetrics.AUROC(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary', threshold=0.5)
        self.val_spec = torchmetrics.Specificity(task='binary', threshold=0.5)
        self.val_sens = torchmetrics.Recall(task='binary', threshold=0.5)
        self.val_auc = torchmetrics.AUROC(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary', threshold=0.5)
        self.test_spec = torchmetrics.Specificity(task='binary', threshold=0.5)
        self.test_sens = torchmetrics.Recall(task='binary', threshold=0.5)
        self.test_auc = torchmetrics.AUROC(task='binary')

        if opt.damper:
            if opt.imagesize < 336:
                self.model_pre = WaveDamper(wavelet='bior2.2', level=4, min_severity=0.8, max_severity=1).cuda()
                checkpoint = torch.load(f'/home/middeljans/COSMO/pretrained/damper_256.ckpt')['state_dict']
            elif opt.imagesize == 336:
                self.model_pre = WaveDamper(wavelet='bior2.2', level=4, min_severity=0.8, max_severity=1).cuda()
                checkpoint = torch.load(f'/home/middeljans/COSMO/pretrained/damper_336.ckpt')['state_dict']
            elif opt.imagesize > 336:
                self.model_pre = WaveDamper(wavelet='bior2.2', level=5, min_severity=0.8, max_severity=1).cuda()
                checkpoint = torch.load(f'/home/middeljans/COSMO/pretrained/damper_512.ckpt')['state_dict']
            checkpoint_keys = list(checkpoint.keys())
            for key in checkpoint_keys:
                checkpoint[key.replace('model.backbone.', '')] = checkpoint[key]
                del checkpoint[key]
            self.model_pre.load_state_dict(checkpoint, strict=True)
            self.model_pre.eval()
            print('Using damper for augmentation!')

    def forward(self, x):

        # Extract outputs of the model
        cls_out = self.model(x)

        return cls_out

    def configure_optimizers(self):

        # Define learning rate
        learning_rate = opt.train_lr

        # Define optimizer
        optimizer = construct_optimizer(optim=opt.optimizer, parameters=self.parameters(), lr=learning_rate)

        # Define learning rate scheduler
        scheduler = construct_scheduler(schedule=opt.scheduler, optimizer=optimizer, lr=learning_rate,
                                        metric='val_loss_cls')

        if scheduler is not None:
            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler}
        else:
            return optimizer

    def training_step(self, train_batch, batch_idx):

        # Extract images, labels
        img, lab = train_batch

        # Damper augmentation
        if opt.damper:
            with torch.no_grad():
                bs = img.size(0)
                indices_1 = torch.randperm(bs, device=img.device)
                split1 = indices_1[bs // 2:]
                damped_x = self.model_pre(img[split1])
                img[split1] = damped_x
                if opt.phase:
                    img = mix_data(img, prob=0.25)

        preds = self.forward(img)

        # Perform label smoothing
        lab_smooth = (1.-self.label_smoothing)*lab + self.label_smoothing*0.5

        # Compute Classification Loss
        cls_loss = self.cls_criterion(preds, lab_smooth)
        self.log('train_loss_cls', cls_loss.item())

        # Update metrics
        logits_cls = self.sigmoid(preds)

        self.train_auc.update(logits_cls, lab.to(torch.int32))

        return cls_loss

    def on_train_epoch_end(self):

        # Compute metrics
        train_auc = self.train_auc.compute()

        # Log and print metric value
        self.log('train_auc', train_auc)
        print('\n' + 120 * "=")
        print(f"Training Set:  AUC Cls: {train_auc:.4}")
        print(120 * "=" + '\n')

        # Reset metric values
        self.train_auc.reset()

    def validation_step(self, val_batch, batch_idx):

        # Extract images, labels
        img, lab = val_batch

        # Extract predictions of the network
        preds = self.forward(img)

        # Perform label smoothing
        lab_smooth = (1. - self.label_smoothing) * lab + self.label_smoothing * 0.5

        # Compute Classification Loss
        cls_loss = self.cls_criterion(preds, lab_smooth)
        self.log('val_loss_cls', cls_loss.item())

        # Update metrics
        logits_cls = self.sigmoid(preds)
        self.val_acc.update(logits_cls, lab.to(torch.int32))
        self.val_sens.update(logits_cls, lab.to(torch.int32))
        self.val_spec.update(logits_cls, lab.to(torch.int32))
        self.val_auc.update(logits_cls, lab.to(torch.int32))

        return cls_loss

    def on_validation_epoch_end(self):

        # Compute metric values
        val_acc = self.val_acc.compute()
        val_sens = self.val_sens.compute()
        val_spec = self.val_spec.compute()
        val_auc = self.val_auc.compute()

        # Log and print values
        self.log('val_acc', val_acc)
        self.log('val_sens', val_sens)
        self.log('val_spec', val_spec)
        self.log('val_auc', val_auc)
        print('\n\n' + 120 * "=")
        print(f"Validation Set: Accuracy: {val_acc:.4}, Sensitivity: {val_sens:.4}, "
              f"Specificity: {val_spec:.4}, AUC Cls: {val_auc:.4}")
        print(120 * "=" + '\n')

        # Reset metric values
        self.val_acc.reset()
        self.val_sens.reset()
        self.val_spec.reset()
        self.val_auc.reset()

    def test_step(self, test_batch, batch_idx):

        # Extract images, labels, mask and has_mask
        img, lab = test_batch

        # Extract predictions of the network
        preds = self.forward(img)

        # Update metrics
        logits_cls = self.sigmoid(preds)
        self.test_acc.update(logits_cls, lab.to(torch.int32))
        self.test_sens.update(logits_cls, lab.to(torch.int32))
        self.test_spec.update(logits_cls, lab.to(torch.int32))
        self.test_auc.update(logits_cls, lab.to(torch.int32))

    def on_test_epoch_end(self):

        # Execute metric computation
        test_acc = self.test_acc.compute()
        test_sens = self.test_sens.compute()
        test_spec = self.test_spec.compute()
        test_auc = self.test_auc.compute()

        # Print results
        print('\n\n' + 120 * "=")
        print(f"Test Set: Accuracy: {test_acc:.4}, Sensitivity: {test_sens:.4}, "
              f"Specificity: {test_spec:.4}, AUC Cls: {test_auc:.4}")
        print(120 * "=" + '\n')

        # Reset metric values
        self.test_acc.reset()
        self.test_sens.reset()
        self.test_spec.reset()
        self.test_auc.reset()


""""""""""""""""""""""""""""""
"""" FUNCTION FOR EXECUTION """
""""""""""""""""""""""""""""""

def run_cls(opt):

    """TEST DEVICE"""
    check_cuda()
    torch.use_deterministic_algorithms(mode=False)#, warn_only=True)

    """SETUP PYTORCH LIGHTNING DATAMODULE"""
    print('Starting PyTorch Lightning DataModule...')
    criteria = get_data_inclusion_criteria(opt)
    data_transforms = augmentations_cls(opt)
    dm_train = WLEDataModule_CLS(data_dir=CACHE_PATH, criteria=criteria, transforms=data_transforms, opt=opt)

    """SETUP PYTORCH LIGHTNING MODEL"""
    print('Starting PyTorch Lightning Model...')

    # Construct Loggers for PyTorch Lightning
    wandb_logger_train = WandbLogger(name='{}'.format(EXPERIMENT_NAME, opt.seed), project='Scope Study ResNet',
                                     save_dir=os.path.join(SAVE_DIR, EXPERIMENT_NAME), offline=False)
    lr_monitor_train = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(monitor='val_auc', min_delta=0.0005, patience=5, mode='max', check_on_train_epoch_end=False)

    # Construct callback used for training the model
    checkpoint_callback_train = ModelCheckpoint(
        monitor='val_auc',
        mode='max',
        dirpath=os.path.join(SAVE_DIR, EXPERIMENT_NAME),
        filename='model-{epoch:02d}-{val_auc:.4f}',
        save_top_k=1,
        save_weights_only=True
    )

    """TRAINING PHASE"""

    # Construct PyTorch Lightning Trainer
    pl_model = WLEModel_CLS(opt=opt)
    trainer = pl.Trainer(devices=1,
                         accelerator="gpu",
                         max_epochs=opt.num_epochs,
                         logger=wandb_logger_train,
                         callbacks=[checkpoint_callback_train,
                                    lr_monitor_train,
                                    early_stop_callback],
                         check_val_every_n_epoch=1,
                         deterministic=False)

    # Start Training
    trainer.fit(model=pl_model, datamodule=dm_train)

    # Finish WandB logging
    wandb_logger_train.experiment.finish()

    """INFERENCE PHASE"""
    best_index = find_best_model(path=os.path.join(SAVE_DIR, EXPERIMENT_NAME))

    criteria = get_data_inclusion_criteria(opt)
    dm_test = WLEDataModule_CLS(data_dir=CACHE_PATH, criteria=criteria, transforms=data_transforms, opt=opt)

    trainer.test(model=pl_model,
                 datamodule=dm_test,
                 ckpt_path=os.path.join(SAVE_DIR, EXPERIMENT_NAME, best_index))
 
""""""""""""""""""""""""""
"""EXECUTION OF FUNCTIONS"""
""""""""""""""""""""""""""

if __name__ == '__main__':

    """SPECIFY PATH FOR SAVING"""
    EXPERIMENT_NAME = 'test'
    SAVE_DIR = os.path.join(os.getcwd(), 'experiments')
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    """SPECIFY CACHE PATH"""
    CACHE_PATH = r'/home/middeljans/COSMO/cache_seg'

    """SPECIFY PARAMETERS AND INCLUSION CRITERIA"""

    for i in range(1):
        opt = get_params()
        EXPERIMENT_NAME = f"{opt.basename}"
        opt.experimentname = EXPERIMENT_NAME

        # Check if direction for logging the information already exists; otherwise make direction
        if not os.path.exists(os.path.join(SAVE_DIR, opt.experimentname)):
            os.mkdir(os.path.join(SAVE_DIR, opt.experimentname))

        # Save params from opt as a dictionary in a json file 'params.json'
        with open(os.path.join(SAVE_DIR,  opt.experimentname, 'params.json'), 'w') as fp:
            json.dump(opt.__dict__, fp, indent=4)

        # Save inclusion criteria (already dictionary) in a json file 'datacriteria.json'
        with open(os.path.join(SAVE_DIR, opt.experimentname, 'datacriteria.json'), 'w') as fp:
            json.dump(get_data_inclusion_criteria(opt), fp, indent=4)

        """EXECUTE FUNCTION"""
        run_cls(opt=opt)
        
        opt.val_fold = [f + 1 for f in opt.val_fold]