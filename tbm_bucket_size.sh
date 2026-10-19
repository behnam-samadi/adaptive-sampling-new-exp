CUDA_VISIBLE_DEVICES=3 python examples/segmentation/main.py --cfg cfgs/s3dis/pointvector-l-2.yaml mode=test --pretrained_path checkpoints/s3dis_pointvector.pth | tee revsion/reviewer2/comment2/s3dis/point_vector/point_vector_proposed_14_bukcet_size.txt


CUDA_VISIBLE_DEVICES=3 python examples/segmentation/main.py --cfg cfgs/s3dis/pointnext-l.yaml mode=test --pretrained_path checkpoints/s3dis_pointnext_l.pth | tee revsion/reviewer2/comment2/s3dis/pointnext_l/pointnext_proposed_14_bucket_size.txt


CUDA_VISIBLE_DEVICES=3 python examples/segmentation/main.py --cfg cfgs/scannet/pointnext-l.yaml mode=test --pretrained_path checkpoints/scan>

