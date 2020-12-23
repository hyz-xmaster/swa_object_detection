_base_ = '../fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py'
model = dict(
    pretrained='open-mmlab://detectron/resnet101_caffe',
    backbone=dict(depth=101))
lr_config = dict(
    _delete_=True,
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=12,
    step_ratio_up=0.0)
total_epochs = 12
