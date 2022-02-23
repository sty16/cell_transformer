# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[20, 40, 60])
runner = dict(type='EpochBasedRunner', max_epochs=80)
