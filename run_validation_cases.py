#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:46:00 2018

@author: David
"""

import os
import json

from unet3d.prediction import run_validation_cases

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)

config_json = os.path.abspath('models/T1/51U_tissue_w_template_init_pennmodel.json')
with open(config_json) as jf:
    config = json.load(jf)

def main():
    prediction_dir = os.path.abspath("predictions/T1/validation")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir,
                         threshold=0.5)


if __name__ == "__main__":
    main()
