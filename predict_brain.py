#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:46:00 2018

@author: jiancongwang
"""

import os
import sys, getopt

from unet3d.prediction import predict_brain_all_modalities
from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators

config = dict()
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = list()
config["labels"].append(1)
config["n_labels"] = len(config["labels"])
config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.

config["batch_size"] = 1
config["validation_split"] = 0  # portion of the data that will be used for training
config["patch_shape"] = None
config["validation_batch_size"] = 2
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

def main(argv):
    try:
      opts, args = getopt.getopt(argv,"hi:o:s:",["idir=","odir=","sid="])
    except getopt.GetoptError:
      print('Usage: predict_brain.py -i <inputdir> -o <outputdir> -s <subjectID>')
      sys.exit(2)
    if len(opts) < 2:
      print('Must specify at least the subject ID and destination of predictions')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('Usage: predict_brain.py -i <inputdir> -o <outputdir> -s <subjectID>')
         sys.exit()
      elif opt in ("-i", "--idir"):
         patientDir = os.path.abspath(arg)
      elif opt in ("-o", "--odir"):
         outputDir = arg
      elif opt in ("-s", "--sid"):
          subjectID = str(arg)


    model_file = os.path.abspath(os.path.join(os.path.dirname(__file__),"models/modality_independent/brain.h5"))
    config["training_file"] = os.path.join(outputDir,subjectID,"training.pkl")
    config["validation_file"] = os.path.join(outputDir,subjectID,"validation.pkl")
    
    modality = ('T1','T2','T1Post','FLAIR','GRE','ADC','SingleDWI','T1Sag')
    modalities = list()
    subjectIDs = list()
    file_tuple = []
    for mod in modality:
        inputfile = os.path.join(patientDir,subjectID,mod,"%s_%s.nii.gz" % (subjectID, mod))
        subject_files = list()
        if os.path.exists(inputfile):
            modalities.append(mod)
            subjectIDs.append(subjectID)
            subject_files.append(inputfile)
            subject_files.append(inputfile)
            file_tuple.append(tuple(subject_files))
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    if not os.path.exists(os.path.join(outputDir,subjectID)):
        os.mkdir(os.path.join(outputDir,subjectID))
        hdf5_file = os.path.join(outputDir,subjectID,"data.h5")

        write_data_to_file(file_tuple, hdf5_file, 
                   image_shape=config["image_shape"],
                       subject_ids=subjectIDs,
                       modality=modalities)
        data_file = open_data_file(hdf5_file)
        
        train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
            data_file,
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
    
        predict_brain_all_modalities(config["validation_file"], outputDir, model_file, hdf5_file, modalities,
                        output_label_map=True, threshold=0.5, labels=config["labels"], overlap=16, permute=False)
        
        for mod in modalities:
            prediction_file = os.path.join(outputDir,subjectID,"prediction_{0}.nii.gz".format(mod))
            inputfile = os.path.join(patientDir,subjectID,mod,"%s_%s.nii.gz" % (subjectID, mod))
            if os.path.exists(prediction_file):
                call = "c3d " + inputfile + " " + prediction_file + \
                    " -int 0 -reslice-identity" + \
                    " -o " + os.path.join(outputDir,subjectID,"prediction_{0}_native.nii.gz".format(mod))
        
        #" -int 0 -reslice-itk " + os.path.join(outputDir,subjectID[0],"identity.txt")
                os.system(call)
            else:
                continue

if __name__ == "__main__":
    main(sys.argv[1:])
