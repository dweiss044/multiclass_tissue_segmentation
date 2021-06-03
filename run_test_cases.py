#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:22:56 2021

@author: David
"""

import os
import csv
import json

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.prediction import run_validation_cases

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)

config_json = os.path.abspath('models/T1/51U_tissue_w_template_init_pennmodel.json')
with open(config_json) as jf:
    config = json.load(jf)

config["data_file"] = os.path.abspath("data/51U_T1_tissue_w_template_test.h5")
config["training_file"] = os.path.abspath("data/T1tissue_w_template_test1.pkl")
config["validation_file"] = os.path.abspath("data/T1tissue_w_template_test2.pkl")
config["validation_split"] = 1  # portion of the data that will be used for training

test_data_files = list()
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

for i in range(len(testDirs)):
    subject_files = list()
    subject_files.append("%s_%s" % (trainDirs[i],"T1.nii.gz"))
    subject_files.append("%s_%s" % (trainDirs[i],"spatial_prior.nii.gz"))
    subject_files.append("%s_%s" % (trainDirs[i],"seg.nii.gz"))
    test_data_files.append(tuple(subject_files))

if not(os.path.exists(config["data_file"])):    
    write_data_to_file(test_data_files, config["data_file"], 
                       image_shape=config["image_shape"],
                       subject_ids=testID, 
                       label_indices=[len(subject_files)-1, len(subject_files)-2])

data_file_opened = open_data_file(config["data_file"])

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


prediction_dir = os.path.abspath("predictions/T1/test")
run_validation_cases(validation_keys_file=config["training_file"],
                     model_file=config["model_file"],
                     training_modalities=config["training_modalities"],
                     labels=config["labels"],
                     hdf5_file=config["data_file"],
                     output_label_map=True,
                     output_dir=prediction_dir,
                     threshold=0.5)


