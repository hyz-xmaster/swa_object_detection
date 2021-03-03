_base_ = ['../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py', '../_base_/swa.py']

# swa optimizer
swa_optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
