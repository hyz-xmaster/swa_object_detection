_base_ = ['../fcos/fcos_r101_caffe_fpn_gn-head_1x_coco.py', '../_base_/swa.py']

# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
