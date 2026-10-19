OUTPUT_DIR="revsion/reviewer1/comment2"
#CHECKPOINT="/mnt/clientshare/samadi/research/DGNet/checkpoints/s3dis_pointnext.pth"
CHECKPOINT="s3dis_pointvector.pth"
CFG="cfgs/s3dis/pointnext-l.yaml"

for B in 14
do
	CUDA_VISIBLE_DEVICES=1 python examples/segmentation/main.py \
	    --cfg $CFG \
	    mode=test \
	    dataset.test.split=test \
	    no_label=True \
	    pretrained_path=$CHECKPOINT \
	    | tee $OUTPUT_DIR/pointnext-s3dis-octree-baseline.txt
done
