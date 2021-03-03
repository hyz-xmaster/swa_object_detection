_base_ = ['../faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py', '../_base_/swa.py']

# swa optimizer
swa_optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
