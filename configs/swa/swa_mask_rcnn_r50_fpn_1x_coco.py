_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
lr_config = dict(
    _delete_=True,
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=12,
    step_ratio_up=0.0)
total_epochs = 12
