#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:34:16 2019

@author: david
"""

import os
import sys, getopt

from unet3d.training import load_old_model
from unet3d.prediction import predict_single_case
from unet3d.data import write_data_to_file, open_data_file

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = dict()
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = list()
config["all_modalities"] = ["1"]
config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.

def main(argv):
    subjectID = []
    
    try:
      opts, args = getopt.getopt(argv,"hi:o:m:s:l:",["ifile=","odir=","mdir=","sid=","maxl="])
    except getopt.GetoptError:
      print('Usage: predict_single_case.py -i <inputfile> -o <outputdir> -m <modelfile> -s <subjectID> -l <nlabels>')
      sys.exit(2)
    if len(opts) < 5:
      print('Usage: predict_single_case.py -i <inputfile> -o <outputdir> -m <modelfile> -s <subjectID> -l <nlabels>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('Usage: predict_single_case.py -i <inputfile> -o <outputdir> -m <modelfile> -s <subjectID> -l <nlabels>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--odir"):
         outputDir = os.path.abspath(arg)
      elif opt in ("-m", "--mdir"):
          task = str(arg).split('/')[-1].split('.')[0]
          model = load_old_model(os.path.abspath(arg))
      elif opt in ("-s", "--sid"):
          subjectID.append(arg)
      elif opt in ("-l","--nlabels"):
          for i in range(int(arg)):
              config["labels"].append(i+1)
    
    file_tuple = []
    subject_files = list()
    subject_files.append(inputfile)
    subject_files.append(inputfile)
    file_tuple.append(tuple(subject_files))
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    if not os.path.exists(os.path.join(outputDir,subjectID[0])):
        os.mkdir(os.path.join(outputDir,subjectID[0]))
        hdf5_file = os.path.join(outputDir,subjectID[0],"data.h5")
        write_data_to_file(file_tuple, hdf5_file, 
                   image_shape=config["image_shape"],
                       subject_ids=subjectID)
        data_file = open_data_file(hdf5_file)
    
        predict_single_case(outputDir, 
                            model, 
                            data_file, 
                            config["all_modalities"],
                            output_label_map=True, 
                            threshold=0.5, 
                            labels=config["labels"], 
                            overlap=16, 
                            permute=False,
                            task=task)
        
        call = "c3d " + inputfile + " " + \
        os.path.join(outputDir,subjectID[0],"prediction.nii.gz") + \
        " -reslice-identity" + \
        " -o " + os.path.join(outputDir,subjectID[0],"prediction_native.nii.gz")
        
        os.system(call)

if __name__ == "__main__":
   main(sys.argv[1:])
