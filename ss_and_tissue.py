import os
import sys, getopt
import argparse
import numpy as np
import nibabel as nib
import scipy.ndimage
import time
import csv

from unet3d.training import load_old_model
from unet3d.prediction import prediction_to_image
from unet3d.data import write_data_to_file, open_data_file

timefile = 'timing/time_test.csv'

cur_dir = os.path.dirname(os.path.realpath(__file__))

config = dict()
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = list()
config["all_modalities"] = ["T1"]
config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.

def skullstrip_and_tissue(output_dir, ss_model, tissue_model, data_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    tic = time.clock()

    affine = data_file.root.affine[0]
    t1_data = np.asarray([data_file.root.data[0]])
    prior_data = np.asarray(data_file.root.truth[0][0])
    
    image = nib.Nifti1Image(t1_data[0, 0], affine)
    image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format("t1")))
    
    prediction = ss_model.predict(t1_data)
    brainmask = np.zeros_like(prediction)
    brainmask[prediction > 0.5] = 1

    brainmask = fill_holes(brainmask, 0.5)
    #brainmask_image = nib.Nifti1Image(brainmask, affine)
    #brainmask_image.to_filename(os.path.join(output_dir, "brainmask.nii.gz"))

    t1ss = np.copy(t1_data).flatten()
    brainmask_flat = brainmask.flatten()
    t1ss[brainmask_flat == 0] = np.min(t1ss)
    t1ss = np.reshape(t1ss,t1_data.shape)
    ss_image = nib.Nifti1Image(np.squeeze(t1ss), affine)
    ss_image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format("t1ss")))

    toc = time.clock()
    time_ss = toc - tic
    
    
    test_data = np.concatenate([t1ss,prior_data[np.newaxis][np.newaxis]], axis=1)
    tic = time.clock()
    prediction = tissue_model.predict(test_data)
    toc = time.clock()
    prediction_image = prediction_to_image(prediction, affine, label_map=True, threshold=0.5,
                                           labels=[1,2,3,4,5,6])
    prediction_image.to_filename(os.path.join(output_dir, "prediction.nii.gz"))
    
    time_tissue = toc - tic
    #prediction_image = prediction_to_image(prediction, affine, label_map=True, threshold=threshold,
    #                                     labels=labels)
    #if isinstance(prediction_image, list):
    #    for i, image in enumerate(prediction_image):      
    #        image.to_filename(os.path.join(case_directory, "prediction_{0}.nii.gz".format(i + 1)))
    #else:
        # prediction_image.to_filename(os.path.join(case_directory, "prediction.nii.gz"))

    return time_ss, time_tissue

def fill_holes(prediction, threshold):
    try:
        print("Filling holes")
        prediction_bin = np.zeros_like(prediction)
        prediction_bin[prediction > threshold] = 1
        prediction_bin = np.squeeze(prediction_bin)
        return scipy.ndimage.morphology.binary_fill_holes(prediction_bin).astype(np.int8)
    except:
        print("Error filling holes, install scipy.")
        return prediction

def main():
    tic = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('t1', type=str, help='T1 image')
    parser.add_argument('prior', default=None, help='Spatial prior image')
    parser.add_argument('outdir',default='./tmp',help='Where to save predictions')
    parser.add_argument('model',default='models/paper/tissue_w_template_NNprior.h5',help='Model to use.')
    parser.add_argument('ssmodel',default='models/modality_independent/brain.h5',help='Skullstripping model to use.')
    parser.add_argument('--gpu', default='1',help='Which gpu to use')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    file_tuple = []
    subject_files = list()
    subject_files.append(args.t1)
    subject_files.append(args.prior)
    #subject_files.append(args.prior)
    file_tuple.append(tuple(subject_files))

    outputDir = './tmp'
    hdf5_file = os.path.join(outputDir,"data.h5")
    write_data_to_file(file_tuple, hdf5_file, 
                       image_shape=config["image_shape"],
                       subject_ids=['unidentified'])#,
                       #label_indices=[len(subject_files)-2, len(subject_files)-1])
    data_file = open_data_file(hdf5_file)

    ss_model = load_old_model(os.path.join(cur_dir, args.ssmodel))
    tissue_model = load_old_model(os.path.join(cur_dir,args.model))

    toc = time.clock()
    time_preprocess = toc - tic
    time_ss, time_tissue = skullstrip_and_tissue(outputDir,ss_model,tissue_model,data_file)

    with open(timefile, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([time_ss,time_tissue,time_preprocess])

if __name__ == '__main__':
    main()
