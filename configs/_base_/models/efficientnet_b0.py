# model settings
pretrained = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientNet',
        arch='b0',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone',)
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
