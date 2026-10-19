CHECKPOINT="/mnt/clientshare/samadi/research/DGNet/checkpoints/s3dis_pointnext_l.pth"
B=14
CFG="/mnt/clientshare/samadi/research/DGNet/cfgs/s3dis/pointnext-l.yaml"
CUDA_VISIBLE_DEVICES=3 bash script/main_segmentation.sh $CFG wandb.use_wandb=False b_val=$B mode=test --pretrained_path $CHECKPOINT 
