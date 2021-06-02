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

config = dict()
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = list()
config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.

cur_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(cur_dir,'models')

def main(argv):
    subjectID = []
    
    try:
      opts, args = getopt.getopt(argv,"hi:t:m:o:s:",["idir=","task=","mod=","odir=","sid="])
    except getopt.GetoptError:
      print('Usage: predict_single_case_v3.py -i <inputDir> -t <task> -m <modality> -o <outputdir> -s <subjectID>')
      sys.exit(2)
    if len(opts) < 4:
      print('Usage: predict_single_case_v3.py -i <inputDir> -t <task> -m <modality> -o <outputdir> -s <subjectID>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('Usage: predict_single_case_v3.py -i <inputDir> -t <task> -m <modality> -o <outputdir> -s <subjectID>')
         sys.exit()
      elif opt in ("-i", "--idir"):
          pDir = True
          patientDir = arg
      elif opt in ("-t", "--task"):
          if arg == 'skullstrip':
              config["labels"].append(1)
              taskh5 = 'brain.h5'
          elif arg == 'tissue':
              taskh5 = 'tissue_w_template_NNprior.h5'
              for i in range(6):
                  config["labels"].append(i+1)
          elif arg == 'lesion':
              config["labels"].append(1)
              taskh5 = 'lesion.h5'
          elif arg == 'enhancement':
              config["all_modalities"] = ["t1","t1ce","diff"]
              taskh5 = 'enhancement.h5'
              config["labels"].append(1)
          elif arg == 'diffusion':
              config["all_modalities"] = ["singledwi","adc"]
              taskh5 = 'diffusion.h5'
              config["labels"].append(1)
          else:
              print('Invalid task.  Specify "skullstrip", "tissue", "lesion", or "enhancement"')
              sys.exit()
      elif opt in ("-o", "--odir"):
         outputDir = os.path.abspath(arg)
      elif opt in ("-m", "--mod"):
          if arg in ("t1","T1"):
              config["all_modalities"] = ["t1"]
              modality = 'T1'
          elif arg in ("T1Sag","t1sag","T1sag"):
              config["all_modalities"] = ["t1"]
              modality = 'T1Sag'
          elif arg in ("t2","T2"):
              config["all_modalities"] = ["t2"]
              modality = 'T2'
          elif arg in ("flair","FLAIR"):
              config["all_modalities"] = ["flair"]
              modality = 'FLAIR'
          elif arg in ("t1post","T1Post","T1post"):
              config["all_modalities"] = ["t1post"]
              modality = 'T1Post'
          elif arg in ("adc","ADC"):
              config["all_modalities"] = ["adc"]
              modality = 'ADC'
          elif arg in ("gre","GRE"):
              config["all_modalities"] = ["gre"]
              modality = 'GRE'
          elif arg in ("singledwi","SingleDWI","singleDWI"):
              config["all_modalities"] = ["singledwi"]
              modality = 'SingleDWI'
          else:
              print('Unacceptable modality')
              sys.exit()
      elif opt in ("-s", "--sid"):
          subjectID.append(arg)
    
    if not pDir:
        patientDir = os.path.abspath('../patients')
    
    file_tuple = []
    subject_files = list()
    label_indices = None
    if taskh5 == 'enhancement.h5':
        t1 = os.path.join(patientDir,'T1Post/%s_T1_to_T1Post.nii.gz' % subjectID[0])
        t1post = os.path.join(patientDir,'T1Post/%s_T1Post.nii.gz' % subjectID[0])
        if not os.path.exists(t1):
            t1 = os.path.join(patientDir,'T1Post/%s_T1Sag_to_T1Post.nii.gz' % subjectID[0])
            t1postdiff = os.path.join(patientDir,'T1Post/%s_T1Post_T1Sag_diff.nii.gz' % subjectID[0])
        else:
            t1postdiff = os.path.join(patientDir,'T1Post/%s_T1Post_T1_diff.nii.gz' % subjectID[0])
        if os.path.exists(t1post):
            if os.path.exists(t1):
                if os.path.exists(t1postdiff):
                    subject_files.append(t1)
                    subject_files.append(t1post)
                    subject_files.append(t1postdiff)
                    subject_files.append(t1postdiff)
                    file_tuple.append(subject_files)
                else:
                    print('T1PostDiff image does not exist!')
            else:
                print('T1 image does not exist!')
        else:
            print('T1Post image does not exist!')
        
        model = load_old_model(os.path.join(model_dir,'combined',taskh5))
    elif taskh5 == 'diffusion.h5':
        singledwi = os.path.join(patientDir,'SingleDWI/%s_SingleDWI.nii.gz' % subjectID[0])
        adc = os.path.join(patientDir,'ADC/%s_ADC.nii.gz' % subjectID[0])
        if os.path.exists(singledwi):
            if os.path.exists(adc):
                subject_files.append(adc)
                subject_files.append(singledwi)
                subject_files.append(singledwi)
                file_tuple.append(subject_files)
            else:
                print('ADC image does not exist!')
        else:
            print('SingleDWI image does not exist!')
        
        model = load_old_model(os.path.join(model_dir,'combined',taskh5))
    elif taskh5 == 'brain.h5':
        inputfile = os.path.join(patientDir,subjectID[0],modality,'%s_%s.nii.gz' % (subjectID[0],modality))
        subject_files.append(inputfile)
        subject_files.append(inputfile)
        file_tuple.append(tuple(subject_files))
        if modality == "T1Sag":
            model = load_old_model(os.path.join(model_dir,'T1',taskh5))
        else:
            model = load_old_model(os.path.join(model_dir,modality,taskh5))
    elif taskh5 == 'tissue_w_template_NNprior.h5':
        config["all_modalities"] = ["t1","prior"]
        inputfile = os.path.join(patientDir,subjectID[0],modality,'%s_%s_skullstripped_trim_upsampled.nii.gz' % (subjectID[0],modality))
        template_seg = os.path.join(patientDir,subjectID[0],modality,'TemplateReg/template_to_%s_%s_trim_greedy_TissueSeg.nii.gz' % (subjectID[0],modality))
        subject_files.append(inputfile)
        subject_files.append(template_seg)
        subject_files.append(template_seg)
        file_tuple.append(tuple(subject_files))
        label_indices = [len(subject_files) - 2, len(subject_files) - 1]
        if modality == "T1Sag":
            model = load_old_model(os.path.join(model_dir,'T1',taskh5))
        else:
            model = load_old_model(os.path.join(model_dir,modality,taskh5))
    else:
        inputfile = os.path.join(patientDir,subjectID[0],modality,'%s_%s_skullstripped.nii.gz' % (subjectID[0],modality))
        subject_files.append(inputfile)
        subject_files.append(inputfile)
        file_tuple.append(tuple(subject_files))
        if modality == "T1Sag":
            model = load_old_model(os.path.join(model_dir,'T1',taskh5))
        else:
            model = load_old_model(os.path.join(model_dir,modality,taskh5))
        
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    if not os.path.exists(os.path.join(outputDir,subjectID[0])):
        os.makedirs(os.path.join(outputDir,subjectID[0]))
    hdf5_file = os.path.join(outputDir,subjectID[0],"data.h5")
    write_data_to_file(file_tuple, hdf5_file, 
                   image_shape=config["image_shape"],
                       subject_ids=subjectID,
                       label_indices = label_indices)
    data_file = open_data_file(hdf5_file)
    
    if taskh5 in ['enhancement.h5', 'diffusion.h5']: 
        predict_single_case(outputDir, model, data_file, config["all_modalities"],
                        output_label_map=False, labels=config["labels"], overlap=16, permute=False)
        if taskh5 == 'enhancement.h5':
            reference = t1post
        elif taskh5 == 'diffusion.h5':
            reference = singledwi
        call = "c3d " + reference + " " + os.path.join(outputDir,subjectID[0],"prediction.nii.gz") + \
            " -int 1 -reslice-identity" + \
            " -o " + os.path.join(outputDir,subjectID[0],"prediction_native.nii.gz")
    else:
        predict_single_case(outputDir, model, data_file, config["all_modalities"],
                        output_label_map=True, threshold=0.5, labels=config["labels"], overlap=16, permute=False)
        call = "c3d " + inputfile + " " + os.path.join(outputDir,subjectID[0],"prediction.nii.gz") + \
            " -int 0 -reslice-identity " + \
            "-o " + os.path.join(outputDir,subjectID[0],"prediction_native.nii.gz")
        
    os.system(call)

if __name__ == "__main__":
   main(sys.argv[1:])
