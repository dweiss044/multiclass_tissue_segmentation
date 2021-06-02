# Pre-requisites

The easiest way to use this network is with anaconda.  Once anaconda is installed,
downloading the installing the environment from the unet-cpu.yml file will
install and set up most of the dependencies.  Some pip versions cannot find the
keras-contrib package so I have removed it from the .yml file and we will have to
install it directly from git after installing the conda environment.  First,
install the conda environment from the .yml file with,

```
$ conda env create -f unet-cpu.yml
```

If there are any packages that cannot be found feel free to remove them from
the file unet-cpu.yml and manually install them with conda or pip.  Next we must
install the keras-contrib package from github:
```
$ pip install git+https://www.github.com/keras-team/keras-contrib.git
```
The final pre-requisite is the convert3d tool provided by itksnap.  This can be
downloaded at

http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.C3D

# Skull stripping

The python program for predicting is predict_single_case.py.  Help for this
program is available by typing
```
$ python predict_single_case.py -h
```
However, there are two skullstripping wrappers provided:
 
1) skullstrip_image.sh is used for skullstripping a single image. This script
assumes no directory structure and thus is much easier to use. We just need to
tell it which file to skullstrip and give it a unique identifier so that the
output filename is unique.  It is easiest to feed it the accession number.
This program has three possible inputs,
   
   i) input filename (required)
  ii) Unique identifying [Accession] number (required)
 iii) output filename (optional)
 
If the output file is not specified the output will be saved in the present working
directory with the naming convention ${ACCESSION}_${MODALITY}_skullstripped.nii.gz.
An example call might be
```
$ ./skullstrip_image.sh ../patients/9146424/FLAIR/FLAIR.nii.gz 9146424 ../patients/9146424/FLAIR/FLAIR_skullstripped.nii.gz
```

2) skullstrip.sh takes as input the base directory where studies can be found,
assuming file naming convention ${BASEDIR}/${ACCESSION}/${MODALITY}/${MODALITY}.nii.gz.
It will skull strip the modality specified in the script (set to FLAIR now) for all
patients in that directory. An example call may be:
```
$ ./skullstrip.sh ../patients
```
where the folder ../patients holds subdirectories for studies according to the
organization ${ACCESSION}/${MODALITY}/${MODALITY}.nii.gz

Models for every modality are included. In order to skull strip a different modality
simply change the MOD variable in the second line of either of the two bash scripts
from FLAIR to the desired modality and use as before.


# Training

A sample training script is provided in train_skullstrip.py.  The config dictionary in
that script holds all the training parameters.  The only real change you'll need to do
to get this program to train on your data is update the read_split function to be able
to find image-segmentation pairs on your hard-drive.  The images are read, cropped,
interpolated, standardized, and serialized into an hdf5 file for streaming data into
GPU RAM during training.  