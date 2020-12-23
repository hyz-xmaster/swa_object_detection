_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
lr_config = dict(
    _delete_=True,
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=24,
    step_ratio_up=0.0)
total_epochs = 24
