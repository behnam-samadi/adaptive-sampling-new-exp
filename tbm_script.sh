#!/bin/bash

# Configuration variables
OUTPUT_DIR="revsion/reviewer1/comment1"
#CHECKPOINT="/mnt/clientshare/samadi/research/DGNet/research/DGNet/log/scannet/scannet-train-pointnext-l-ngpus1-seed1977-20260610-075643-GmrrXZDaoV85vkYZG7A32B/checkpoint/model_best.pth"
#CHECKPOINT="/mnt/clientshare/samadi/research/DGNet/checkpoints/scannet_pointnext_63acc.pth"
CHECKPOINT="/mnt/clientshare/samadi/research/DGNet/checkpoints/s3dis_pointnext_l.pth"
CFG="/mnt/clientshare/samadi/research/DGNet/cfgs/s3dis/pointnext-l.yaml"
#CFG="cfgs/scannet/pointnext-l.yaml"

mkdir -p $OUTPUT_DIR

# Loop through values
for B in 24 28
do
    echo "Running with B=$B"

    # We pass b_val=$B directly as a Hydra override
    #CUDA_VISIBLE_DEVICES=3 python examples/segmentation/main.py \
        #--cfg $CFG \
        #mode=test \
        #dataset.test.split=test \
        #no_label=True \
        #pretrained_path=$CHECKPOINT \
        #b_val=$B \
        #| tee $OUTPUT_DIR/pointnext-s3dis-baseline-B-${B}.txt
        
        CUDA_VISIBLE_DEVICES=3 bash script/main_segmentation.sh $CFG wandb.use_wandb=False b_val=$B mode=test --pretrained_path $CHECKPOINT | tee $OUTPUT_DIR/pointnext-s3dis-proposed_whole_dataset-B-${B}.txt

        
done
#pointnext-scannet-proposed-B-${B}.txt
