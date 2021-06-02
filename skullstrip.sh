#!/bin/bash
MOD=FLAIR
GPU=0
BASEDIR=$1

for sub in $BASEDIR/*; do
    echo $(echo $sub | rev | cut -f 1 -d / | rev) >> ${MOD}.txt
done

# Set up and activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate unet-cpu
mkdir -p predictions/${MOD}

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=${GPU}

for i in $(cat ${MOD}.txt)
do
    IMAGE=$BASEDIR/${i}/${MOD}/${MOD}.nii.gz
    python predict_single_case.py \
	   -i $IMAGE\
	   -o predictions/${MOD}/brain \
	   -m models/${MOD}/brain.h5 \
	   -s ${i} \
	   -l 1
    
    # Select only the largest connected component
    PREDNATIVE=predictions/${MOD}/brain/${i}/prediction_native.nii.gz
    c3d \
	$PREDNATIVE -popas S \
	-push S -thresh 1 inf 1 0 -comp -popas C \
	-push C -thresh 1 1 1 0 \
	-push S -multiply \
	-o $PREDNATIVE

    # Dilate the mask
    c3d \
	$PREDNATIVE -dialte 1 2x2x2vox \
	-o $PREDNATIVE
    
    # Mask the original image with the predicted binary mask
    c3d $IMAGE \
	$IMAGE $PREDNATIVE \
	-multiply \
	$BASEDIR/${i}/${MOD}/${MOD}_skullstripped.nii.gz;

done

conda deactivate
rm ${MOD}.txt
