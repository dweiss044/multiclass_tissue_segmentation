function main()
{
    image=$1
    model=$2
    identifier=$3
    PREDDIR=${4:-prediction}
    register_to_template $image

    python ss_and_tissue.py tmp/T1.nii.gz tmp/spatial_prior.nii.gz tmp $model --gpu 0
    
    mkdir -p $PREDDIR
    # Resample to native space
    greedy -d 3 \
	   -rf tmp/T1.nii.gz \
	   -ri LABEL 0.2vox \
	   -rm tmp/prediction.nii.gz \
	   $PREDDIR/${identifier}_prediction.nii.gz \
	   -r
    
}

function register_to_template()
{
  TMPDIR=./tmp
  if [[ -d $TMPDIR ]]; then
    rm -rf $TMPDIR
  fi
  mkdir -p $TMPDIR

  T1=$1

  cp $T1 $TMPDIR/T1.nii.gz

  OUTDIR=$TMPDIR/TemplateReg
  mkdir -p $OUTDIR
  OUTPREFIX=$OUTDIR/template_to_T1
  INVOUTPREFIX=$OUTDIR/${id}_to_template

  # template images
  TEMPLATEDIR=/home/shared/demo_brain_pipeline/BrainTemplate/OASIS/
  TEMPLATE=$TEMPLATEDIR/T_template0.nii.gz
  TEMPLATEMASK=$TEMPLATEDIR/T_template0_BrainCerebellumMask.nii.gz

  # Init rigid registration
  greedy -d 3 -a -dof 6 -m NCC 2x2x2 \
    -i $T1 $TEMPLATE \
    -o $TMPDIR/greedy_rigid_init.mat \
    -n 400x0x0x0 \
    -ia-image-centers -search 400 5 5

  # perform affine with initialization
  greedy -d 3 -a -dof 12 -m NCC 2x2x2 \
    -n 100x60x20x0 \
    -i $T1 $TEMPLATE \
    -ia $TMPDIR/greedy_rigid_init.mat \
    -o ${OUTPREFIX}_greedy_affine.mat

  # perform deformable registration

  # resample T1 scan into 1x1x1mm3 resolution for deformable registration
  #TMPDIR=$SUBJDIR/$MOD/TemplateReg

  c3d $T1 \
    -resample-mm 1x1x1mm \
    -o $TMPDIR/T1_1x1x1.nii.gz \
    -clear \
    $TEMPLATEMASK \
    -dilate 1 10x10x10vox \
    -o $TMPDIR/template_mask.nii.gz

  greedy -d 3 \
    -r ${OUTPREFIX}_greedy_affine.mat \
    -rf $TMPDIR/T1_1x1x1.nii.gz \
    -ri LABEL 0.2vox \
    -rm $TMPDIR/template_mask.nii.gz \
        $TMPDIR/greedy_affine_mask.nii.gz

  greedy -d 3 -m NCC 2x2x2 -e 0.5 \
    -n 200x100x50x10 \
    -it ${OUTPREFIX}_greedy_affine.mat \
    -i $TMPDIR/T1_1x1x1.nii.gz $TEMPLATE \
    -o ${OUTPREFIX}_greedy_warp.nii.gz \
    -gm $TMPDIR/greedy_affine_mask.nii.gz

  # warp template images and segmentations to T1 space
<<COMMENT
  $GREEDYPATH/greedy -d 3 \
    -r ${OUTPREFIX}_greedy_warp.nii.gz \
        ${OUTPREFIX}_greedy_affine.mat \
    -rf $T1TRIM \
    -rm $TEMPLATE \
        ${OUTPREFIX}_trim_greedy_deformable.nii.gz
COMMENT


  greedy -d 3 \
    -r ${OUTPREFIX}_greedy_warp.nii.gz \
        ${OUTPREFIX}_greedy_affine.mat \
    -rf $T1 \
    -ri LABEL 0.2vox \
    -rm $TEMPLATEDIR/Priors/prior.nii.gz \
        $TMPDIR/spatial_prior.nii.gz \

	
}


main $1 $2 $3 $4
