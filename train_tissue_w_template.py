#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 18:59:28 2018
@author: jiancongwang
"""
#%%
import numpy as np
import os
import glob
import csv
import json

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model, train_model
#from unet3d.metrics import weighted_cce, w_categorical_crossentropy_loss, categorical_crossentropy_loss, w_categorical_crossentropy_old

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)

config = dict()
config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = (1,2,3,4,5,6)  # the label numbers on the input image
config["n_base_filters"] = 16
config["n_labels"] = len(config["labels"])
# Rachit
#config["n_labels"] = len(config["labels"])

config["all_modalities"] = ["t1", "prior"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config["batch_size"] = 1
config["validation_batch_size"] = 2
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 5e-4
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.85  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = True  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["split_file"] = os.path.abspath('split.csv') # File that contains identifiers and train/test split
config["image_dir"] = os.path.abspath('images') # Directory that contains source images and segmentations

config["LOAD_MODEL"] = True # If True then will start training from config["initmodel"]
config["initmodel"] = os.path.abspath('models/T1/tissue_w_template_NNprior.h5')
config["data_file"] = os.path.abspath("data/51U_T1_tissue_w_template.h5")
config["model_file"] = os.path.abspath("models/T1/51U_tissue_w_template_init_pennmodel.h5")
config["training_file"] = os.path.abspath("data/T1tissue_w_template_training.pkl")
config["validation_file"] = os.path.abspath("data/T1tissue_w_template_validation.pkl")
config["overwrite"] = False # If True, will previous files. If False, will use previously written files.

jsonfile = os.path.splitext(config["model_file"])[0]+'.json'
j = json.dumps(config, indent=4)
with open(jsonfile, 'w') as fp:
    print(j, file=fp)

training_data_files = list()
subject_ids = list()

def read_split(splitFile):
    trainID = []
    testID = []
    with open(splitFile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1] == "Train":
                trainID.append(row[0])
            elif row[1] == "Test":
                testID.append(row[0])
    return trainID, testID


trainID, testID = read_split(config["split_file"])

baseDir = config["image_dir"]

trainDirs = ["%s/%s" % (baseDir, x) for x in trainID]
testDirs = ["%s/%s" % (baseDir, x) for x in testID]

for i in range(len(trainDirs)):
    subject_files = list()
    subject_files.append("%s_%s" % (trainDirs[i],"T1.nii.gz"))
    subject_files.append("%s_%s" % (trainDirs[i],"spatial_prior.nii.gz"))
    subject_files.append("%s_%s" % (trainDirs[i],"seg.nii.gz"))
    training_data_files.append(tuple(subject_files))

if not(os.path.exists(config["data_file"])):    
    write_data_to_file(training_data_files, config["data_file"], 
                       image_shape=config["image_shape"],
                       subject_ids=trainID, 
                       label_indices=[len(subject_files)-1, len(subject_files)-2])

data_file_opened = open_data_file(config["data_file"])

if config["LOAD_MODEL"]:
    print("Initializing model from %s" % config["initmodel"])
    model = load_old_model(config["initmodel"])
else:
    model = isensee2017_model(input_shape=config["input_shape"], n_labels=config["n_labels"],
                                                            initial_learning_rate=config["initial_learning_rate"],
                                                            n_base_filters=config["n_base_filters"])
train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=False,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])


train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])

