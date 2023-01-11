# -*- coding: utf-8 -*-
# @Time    : 09/10/2021 16:19
# @Author  : Linhui Dai
# @FileName: deformable_detr_r50_16x2_50e_dota.py.py
# @Software: PyCharm
angle_version = 'oc'
_base_ = [
    '../_base_/datasets/arirang.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='RotatedDeformableDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='/data/2_data_server/cv-01/ao2_a6000/backbone/dota_back.pth')),
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='RotatedDeformableDETRHead',
        num_query=100,
        num_classes=15,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        # frm_cfgs=[
        #     dict(in_channels=256, featmap_strides=[128, 64, 32, 16]),
        #     # dict(in_channels=256, featmap_strides=[128, 64, 32, 16])
        # ],
        transformer=dict(
            type='RotatedDeformableDetrTransformer',
            # use_dab=True,
            # high_dim_query_update = True,
            two_stage_num_proposals=100,
            mixed_selection = True,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='RotatedDeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                # use_dab=True,
                # high_dim_query_update = True,
                # embed_dims = 256,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_points=4
                            )
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=8.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        # loss_bbox=dict(type='RBBoxL1Cost', loss_weight=5.0),
        # loss_bbox=dict('SmoothL1Loss', beta=1.0, loss_weight=1.0),
        reg_decoded_bbox=True,
        # loss_iou=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)
        loss_iou=dict(type='RotatedIoULoss', loss_weight=2.0),
        # loss_iou=dict(type='KFLoss', weight=8.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='Rotated_HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=8.0),
            reg_cost=dict(type='RBBoxL1Cost', weight=5.0, box_format='xywha'),
            # iou_cost=dict(type='KFLossCost', weight=8.0)
            iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=2.0)
            # iou_cost=dict(type='GaussianIoUCost', iou_mode='iou', weight=5.0)
        )),
    test_cfg=dict()
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(512, 512)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(pipeline=train_pipeline, filter_empty_gt=False, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-4,
    # lr = 5e-4,
    # lr= 0.001,
    # momentum=0.9,
    weight_decay=0.0001,
    # weight_decay=0.00005,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
evaluation = dict(interval=1, metric='mAP')
# learning policy
lr_config = dict(policy='step',  warmup_iters=500,warmup_ratio=1.0 / 3,step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=2)
find_unused_parameters = True
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/imp_clsmaxbbox_trainval_300_detFalse/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/imp_deformabledetr_clsmax_bboxtest_300_detFalse/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/imp_frm_base_modify_train_300_detFalse/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/2frmtopbox_p5_baseline/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/proposal_4_300_cossim_iou_twice_test/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/imp_4_300_baseline/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/proposal_4_300_600_cossim2_iou_querymemory_twice_test/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/proposal_4_300_600_cossim2_iou_querycls_twice_test/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/imp_4_300_baseline+clsquery/' np5
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/imp_4_300_baseline_iou_clsquery/' np5
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/imp_4_300_baseline_900iou_clsquery/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/imp_4_300_clsmax_aligndeform_cov3/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/imp_4_300_clsmax_aligndeform_cov3_clsquery/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/imp_4_300_baseline_refangle_8/'

# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/imp_4_300_dytopk_clsquery_trainval_randseed_point5/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/work_dirs/angle_test/'


# work_dir = '/data/2_data_server/cv-01/ao2_a6000/arirang_dir/wh_4_300_dytopk_clsquery_mixedquery_point5_lrbig/'
# work_dir = '/data/2_data_server/cv-01/ao2_a6000/arirang_dir/baseline_4_300/'

# work_dir = '/data/2_data_server/cv-01/ao2_a6000/arirang_dir/100_wh_4_300_dytopksmall_mixedquery_point5/'

# work_dir = '/data/2_data_server/cv-01/ao2_a6000/arirang_dir/label_test_cls15_loss_weight_all_1/'

# work_dir = '/data/2_data_server/cv-01/ao2_a6000/arirang_dir/arirang_hyper_test/'

work_dir = '/data/2_data_server/cv-01/ao2_a6000/arirang_dir/dota_back_mix_cls_852_test/'
