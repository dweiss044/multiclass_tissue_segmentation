import os

import nibabel as nib
import numpy as np
import tables
import time
import cv2
import csv
import SimpleITK as sitk
import tqdm
import scipy
import scipy.ndimage

from .training import load_old_model
from .utils import pickle_load
from .utils.patches import reconstruct_from_patches, get_patch_from_3d_data, compute_patch_indices
from .augment import permute_data, generate_permutation_keys, reverse_permute_data


def patch_wise_prediction(model, data, overlap=0, batch_size=1, permute=False):
    """
    :param batch_size:
    :param model:
    :param data:
    :param overlap:
    :return:
    """
    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    predictions = list()
    indices = compute_patch_indices(data.shape[-3:], patch_size=patch_shape, overlap=overlap)
    batch = list()
    i = 0
    while i < len(indices):
        while len(batch) < batch_size:
            patch = get_patch_from_3d_data(data[0], patch_shape=patch_shape, patch_index=indices[i])
            batch.append(patch)
            i += 1
        prediction = predict(model, np.asarray(batch), permute=permute)
        batch = list()
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape = [int(model.output.shape[1])] + list(data.shape[-3:])
    return reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)


def get_prediction_labels(prediction, threshold=0.5, labels=None):
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0
        if labels:
            for value in np.unique(label_data).tolist()[1:]:
                label_data[label_data == value] = labels[value - 1]
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays


def get_test_indices(testing_file):
    return pickle_load(testing_file)


def predict_from_data_file(model, open_data_file, index):
    return model.predict(open_data_file.root.data[index])


def predict_and_get_image(model, data, affine):
    return nib.Nifti1Image(model.predict(data)[0, 0], affine)


def predict_from_data_file_and_get_image(model, open_data_file, index):
    return predict_and_get_image(model, open_data_file.root.data[index], open_data_file.root.affine)


def predict_from_data_file_and_write_image(model, open_data_file, index, out_file):
    image = predict_from_data_file_and_get_image(model, open_data_file, index)
    image.to_filename(out_file)


def prediction_to_image(prediction, affine, label_map=False, threshold=0.5, labels=None, fill=False):
    if prediction.shape[1] == 1:
        data = prediction[0, 0]
        if label_map:
            label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
    elif prediction.shape[1] > 1:
        if label_map:
            label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
            data = label_map_data[0]
        else:
            return multi_class_prediction(prediction, affine)
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))
    if fill:
        data = fill_holes(data,threshold)
    return nib.Nifti1Image(data, affine)


def multi_class_prediction(prediction, affine):
    prediction_images = []
    for i in range(prediction.shape[1]):
        prediction_images.append(nib.Nifti1Image(prediction[0, i], affine))
    return prediction_images


def run_validation_case(data_index, output_dir, model, data_file, training_modalities,
                        output_label_map=False, threshold=0.5, labels=None, overlap=16, permute=False, 
                        resample_native=False, copyManSeg=False):
    """
    Runs a test case and writes predicted images to file.
    :param data_index: Index from of the list of test cases to get an image prediction from.
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is 
    considered a positive result and will be assigned a label.  
    :param labels:
    :param training_modalities:
    :param data_file:
    :param model:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    affine = data_file.root.affine[data_index]
    test_data = np.asarray([data_file.root.data[data_index]])
    for i, modality in enumerate(training_modalities):
        image = nib.Nifti1Image(test_data[0, i], affine)
        image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))

    test_truth = nib.Nifti1Image(data_file.root.truth[data_index][0], affine)
    test_truth.to_filename(os.path.join(output_dir, "truth.nii.gz"))

    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    if patch_shape == test_data.shape[-3:]:
        prediction = predict(model, test_data, permute=permute)
    else:
        prediction = patch_wise_prediction(model=model, data=test_data, overlap=overlap, permute=permute)[np.newaxis]
    prediction_image = prediction_to_image(prediction, affine, label_map=output_label_map, threshold=threshold,
                                           labels=labels)
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            ofile = os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1))
            image.to_filename(ofile)
    else:
        ofile = os.path.join(output_dir, "prediction.nii.gz")
        prediction_image.to_filename(ofile)
    
    if resample_native:
        case = output_dir.split('/')[-1]
        if not os.path.exists(os.path.join(output_dir,'../native')):
            os.makedirs(os.path.join(output_dir,'../native'))
        native_image = '/media/rnd/Auto_Diag/patients/%s/FLAIR/FLAIR_skullstripped.nii.gz' % case
        call = "c3d " + native_image + " " + ofile + \
            " -reslice-identity" + \
            " -o " + os.path.join(output_dir,'../native/%s.nii.gz' % case)
        os.system(call)
    if copyManSeg:
        case = output_dir.split('/')[-1]
        if not os.path.exists(os.path.join(output_dir,'../ManSeg')):
            os.makedirs(os.path.join(output_dir,'../ManSeg'))
        native_seg = '/media/rnd/Auto_Diag/patients/%s/FLAIR/ManSeg.nii.gz' % case
        ofile = os.path.join(output_dir,'../native/%s_ManSeg.nii.gz' % case)
        call = "cp " + native_seg + " " + ofile
        os.system(call)


def run_validation_cases(validation_keys_file, model_file, training_modalities, labels, hdf5_file,
                         output_label_map=False, output_dir=".", threshold=0.5, overlap=16, permute=False, 
                         resample_native=False, copyManSeg=False):
    validation_indices = pickle_load(validation_keys_file)
    model = load_old_model(model_file)
    data_file = tables.open_file(hdf5_file, "r")
    TIME = []
    for index in tqdm.tqdm(validation_indices):
        if 'subject_ids' in data_file.root:
            case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
        else:
            case_directory = os.path.join(output_dir, "validation_case_{}".format(index))
        start = time.time()
        run_validation_case(data_index=index, output_dir=case_directory, model=model, data_file=data_file,
                            training_modalities=training_modalities, output_label_map=output_label_map, labels=labels,
                            threshold=threshold, overlap=overlap, permute=permute,resample_native=resample_native)
        stop = time.time()
        
        TIME.append(stop-start)
    
    data_file.close()
    with open('TIME.txt', 'w') as f:
        for item in TIME:
            f.write("%s\n" % item)


def predict(model, data, permute=False):
    if permute:
        predictions = list()
        for batch_index in range(data.shape[0]):
            predictions.append(predict_with_permutations(model, data[batch_index]))
        return np.asarray(predictions)
    else:
        return model.predict(data)


def predict_with_permutations(model, data):
    predictions = list()
    for permutation_key in generate_permutation_keys():
        temp_data = permute_data(data, permutation_key)[np.newaxis]
        predictions.append(reverse_permute_data(model.predict(temp_data)[0], permutation_key))
    return np.mean(predictions, axis=0)

## DAVID
def fill_holes_2d(prediction, threshold):
    prediction_bin = (prediction > threshold)
    im_floodfill = prediction_bin.copy()
    h, w = prediction_bin.shape[2:]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill,mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    return (prediction_bin | im_floodfill_inv)

## DAVID
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
        
## DAVID
def predict_single_case(output_dir, model, data_file, training_modalities,
                        output_label_map=False, threshold=0.5, labels=None,
                        overlap=16, permute=False, task=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    affine = data_file.root.affine[0]
    test_data = np.asarray([data_file.root.data[0]])
    case_directory = os.path.join(output_dir, data_file.root.subject_ids[0].decode('utf-8'))
    if not os.path.exists(case_directory):
        os.makedirs(case_directory)
    
    for i, modality in enumerate(training_modalities):
        image = nib.Nifti1Image(test_data[0, i], affine)
        image.to_filename(os.path.join(case_directory, "data_{0}.nii.gz".format(modality)))
    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    if patch_shape == test_data.shape[-3:]:
        prediction = predict(model, test_data, permute=permute)
    else:
        prediction = patch_wise_prediction(model=model, data=test_data, overlap=overlap, permute=permute)[np.newaxis]

    # Fill holes in binary brain mask
    fill = True if task == 'brain' else False
    
    prediction_image = prediction_to_image(prediction, affine, label_map=output_label_map, threshold=threshold,
                                           labels=labels, fill=fill)
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):      
            image.to_filename(os.path.join(case_directory, "prediction_{0}.nii.gz".format(i + 1)))
    else:
        prediction_image.to_filename(os.path.join(case_directory, "prediction.nii.gz"))
        
        
def predict_brain_all_modalities(validation_keys_file,output_dir, model_file, hdf5_file, modalities,
                        output_label_map=False, threshold=0.5, labels=None, overlap=16, permute=False):
    
    validation_indices = pickle_load(validation_keys_file)
    model = load_old_model(model_file)
    data_file = tables.open_file(hdf5_file, "r")
    cnt = 0
    for index in validation_indices:
        #modality = list()
        #modality.append(modalities[cnt])
        modality = list()
        if 'subject_ids' in data_file.root:
            case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
        else:
            case_directory = os.path.join(output_dir, "validation_case_{}".format(index))
        if 'modality' in data_file.root:
            modality.append(data_file.root.modality[index].decode('utf-8'))
        else:
            modality.append(str(cnt))
        #print(case_directory)
        #print(modality)
        run_validation_case_v2(data_index=index, output_dir=case_directory, model=model, data_file=data_file,
                            training_modalities=modality, output_label_map=output_label_map, labels=labels,
                            threshold=threshold, overlap=overlap, permute=permute, task='brain')
        cnt = cnt + 1
    
    data_file.close()
    
def run_validation_case_v2(data_index, output_dir, model, data_file, training_modalities,
                        output_label_map=False, threshold=0.5, labels=None, overlap=16, permute=False, task=None):
    """
    Runs a test case and writes predicted images to file.
    :param data_index: Index from of the list of test cases to get an image prediction from.
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is 
    considered a positive result and will be assigned a label.  
    :param labels:
    :param training_modalities:
    :param data_file:
    :param model:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    affine = data_file.root.affine[data_index]
    test_data = np.asarray([data_file.root.data[data_index]])
    for i, modality in enumerate(training_modalities):
        image = nib.Nifti1Image(test_data[0, i], affine)
        image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))

    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    if patch_shape == test_data.shape[-3:]:
        prediction = predict(model, test_data, permute=permute)
    else:
        prediction = patch_wise_prediction(model=model, data=test_data, overlap=overlap, permute=permute)[np.newaxis]
    # Fill holes in binary brain mask
    fill = True if task == 'brain' else False
    prediction_image = prediction_to_image(prediction, affine, label_map=output_label_map, threshold=threshold, labels=labels, fill=fill)
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            image.to_filename(os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1)))
    else:
        prediction_image.to_filename(os.path.join(output_dir, "prediction_{0}.nii.gz".format(modality)))
