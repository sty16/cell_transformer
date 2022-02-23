# model settings
pretrained_384 = 'https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
pretrained_224 = 'https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_224,
            prefix='backbone',
        ),
    ),
    neck=None,
    head=dict(
    type='VisionTransformerClsHead',
        num_classes=11,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision')),
)
