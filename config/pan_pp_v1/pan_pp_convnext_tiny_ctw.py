model = dict(
    type='PAN_PP',
    backbone=dict(
        type='convnext_tiny',
        pretrained=True
    ),
    neck=dict(
        type='FPEM_v2',
        in_channels=(96, 192, 384, 768),
        out_channels=128
    ),
    detection_head=dict(
        type='PAN_PP_DetHead',
        in_channels=512,
        hidden_dim=128,
        num_classes=6,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_emb=dict(
            type='EmbLoss_v2',
            feature_dim=4,
            loss_weight=0.25
        ),
        use_coordconv=False,
    )
)
data = dict(
    batch_size=4,
    train=dict(
        type='PAN_CUSTOM_V2',
        split='train',
        is_transform=True,
        img_size=640,
        short_size=640,
        kernel_scale=0.7,
        read_type='cv2',
        random_crop_padding_method="PAN"
    ),
    test=dict(
        type='PAN_CUSTOM_V2',
        split='test',
        short_size=640,
        read_type='cv2'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=600,
    optimizer='Adam',
    use_ex=False,
)
test_cfg = dict(
    min_score=0.80,  # original value 0.88
    min_area=256,
    min_kernel_area=2.6,
    scale=4,  # 这个和min_area 关联
    bbox_type='poly',
    result_path='outputs/submit_ic15.zip',
)