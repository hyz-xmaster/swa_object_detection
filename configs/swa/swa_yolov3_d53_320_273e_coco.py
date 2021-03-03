_base_ = ['../yolo/yolov3_d53_320_273e_coco.py', '../_base_/swa.py']

# swa optimizer
swa_optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
swa_optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
