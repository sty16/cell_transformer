# model settings
pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNeXt',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        groups=32,
        width_per_group=4,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone',
        ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
