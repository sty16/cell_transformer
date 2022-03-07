_base_ = [
    '../_base_/models/vit-base-p32.py',
    '../_base_/datasets/imagenet_bs64_pil_resize_autoaug.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

model = dict(
    head=dict(hidden_dim=3072),
    train_cfg=dict(
        augments=dict(type='BatchMixup', alpha=0.2, num_classes=10,
                      prob=1.)))
