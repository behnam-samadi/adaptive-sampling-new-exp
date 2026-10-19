OUTPUT_DIR="revsion/reviewer1/comment2"
CHECKPOINT="/mnt/clientshare/samadi/research/DGNet/checkpoints/s3dis_pointnext.pth"
CFG="cfgs/s3dis/pointnext-l.yaml"

for B in 38 40 42 44 46 48
do
	CUDA_VISIBLE_DEVICES=3 python examples/segmentation/main.py \
	    --cfg $CFG \
	    mode=test \
	    dataset.test.split=test \
	    no_label=True \
	    pretrained_path=$CHECKPOINT \
	    b_val=$B \
            | tee $OUTPUT_DIR/pointnext-s3dis-octree-B-${B}.txt
done
#| tee $OUTPUT_DIR/pointnext-s3dis-octree-B-${B}.txt
