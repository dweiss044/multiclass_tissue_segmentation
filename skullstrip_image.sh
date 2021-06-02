#!/bin/bash

while getopts ":i:s:g:d:m:o:" opt; do
    case $opt in
	i) INPUT_FILE="$OPTARG"
	   ;;
	s) SUBJECT_ID="$OPTARG"
	   ;;
	g) GPU="$OPTARG"
	   ;;
	d) DILATION="$OPTARG"
	   ;;
	m) MOD="$OPTARG"
	   ;;
	o) OUTFILE="$OPTARG"
	   ;;
    esac
done

if [[ "$MOD" == "" ]]; then
    echo "Must specify modality with the -m option"
    exit 100
fi
if [[ "SUBJECT_ID" == "" ]]; then
    echo "Must specify subject ID with the -s option"
    exit 101
fi
# Set up and activate environment
#source ~/anaconda3/etc/profile.d/conda.sh
if [[ "$GPU" != "" ]]; then
    #export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    #export CUDA_VISIBLE_DEVICES=$GPU
    source /data/rauschecker1/dweiss/cluster_utils/activate.sh keras
else
    conda activate unet-cpu
fi

mkdir -p predictions/${MOD}

# Make predictions of brain location in image
python predict_single_case.py \
	   -i $INPUT_FILE \
	   -o predictions/${MOD}/brain \
	   -m models/${MOD}/brain.h5 \
	   -s $SUBJECT_ID \
	   -l 1

output=${OUTFILE-${SUBJECT_ID}_${MOD}_skullstripped.nii.gz}
echo $output

# Select only largest connected component
PREDNATIVE=predictions/${MOD}/brain/${SUBJECT_ID}/prediction_native.nii.gz
c3d \
    $PREDNATIVE -popas S \
    -push S -thresh 1 inf 1 0 -comp -popas C \
    -push C -thresh 1 1 1 0 \
    -push S -multiply \
    -o $PREDNATIVE

# Dilate the mask
if [[ "$DILATION" != "" ]]; then
    c3d \
	$PREDNATIVE -dilate 1 $DILATION \
	-o $PREDNATIVE
fi

# Mask the original image with the predicted binary mask
c3d $INPUT_FILE \
    $INPUT_FILE $PREDNATIVE -multiply \
    -o $output

conda deactivate
