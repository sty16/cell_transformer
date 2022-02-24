# model settings
pretrained = 'https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_bn_batch256_imagenet_20210208-7e55cd29.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VGG', depth=16, norm_cfg=dict(type='BN'), num_classes=10,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone',
        ),
    ),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
