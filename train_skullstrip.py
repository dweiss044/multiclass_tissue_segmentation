#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 18:59:28 2018

@author: jiancongwang
"""

import os
import glob
import csv

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model, train_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = dict()
config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = list()  # the label numbers on the input image
config["labels"].append(1)
config["n_base_filters"] = 16
config["n_labels"] = len(config["labels"])
# Rachit
#config["n_labels"] = len(config["labels"])

config["all_modalities"] = ["FLAIR"]
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
config["patience"] = 5  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 40  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 5e-4
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.90  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = True  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["data_file"] = os.path.abspath("data/FLAIR/skullstrip_training.h5")
config["model_file"] = os.path.abspath("models/FLAIR/brain.h5")
config["training_file"] = os.path.abspath("training.pkl")
config["validation_file"] = os.path.abspath("validations.pkl")
config["overwrite"] = False # If True, will previous files. If False, will use previously written files.

if not os.path.isdir(os.path.dirname(config['data_file'])):
    os.makedirs(os.path.dirname(config['data_file']))

training_data_files = list()
subject_ids = list()

def read_split(splitDir = os.path.abspath('splits_SS/FLAIR_split.csv')):
    trainDirs = []
    testDirs = []
    trainID = []
    testID = []
    baseDir = os.path.abspath('/media/rnd/Auto_Diag/patients')
    with open(splitDir, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            if row[2] == "train":
                trainDirs.append(os.path.join(baseDir,row[1]))
                trainID.append(row[1])
            elif row[2] == "test":
                testDirs.append(os.path.join(baseDir,row[1]))
                testID.append(row[1])
    return trainDirs, testDirs, trainID, testID


trainDirs, testDirs, trainID, testID = read_split(splitDir = os.path.abspath('FLAIR_split.csv'))

for i in range(len(trainDirs)):
    subject_files = list()
    subject_files.append(os.path.join(trainDirs[i],"FLAIR/FLAIR.nii.gz"))
    subject_files.append(os.path.join(trainDirs[i],"FLAIR/brainMask.nii.gz"))
    training_data_files.append(tuple(subject_files))

if not(os.path.exists(config["data_file"])):    
    write_data_to_file(training_data_files, config["data_file"], 
                   image_shape=config["image_shape"],
                       subject_ids=trainID)


data_file_opened = open_data_file(config["data_file"])

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
