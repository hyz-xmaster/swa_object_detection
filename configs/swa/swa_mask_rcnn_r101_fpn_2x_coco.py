_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
lr_config = dict(
    _delete_=True,
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=24,
    step_ratio_up=0.0)
total_epochs = 24
# load_from = 'checkpoints/mask_rcnn/' \
#             'mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth'
# work_dir = 'work_dirs/swa_mask_rcnn_r101_fpn_2x_coco'
