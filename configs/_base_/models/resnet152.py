# model settings
pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_batch256_imagenet_20200708-ec25b1f9.pth'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=152,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone',
        )
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
