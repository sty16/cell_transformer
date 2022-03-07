# model settings
pretrained = 'https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=32,
        drop_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone',
        )
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=10,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
