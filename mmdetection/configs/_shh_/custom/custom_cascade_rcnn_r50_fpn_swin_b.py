
# merge configs
_base_ = [
	# '../models/cascade_rcnn_r50_fpn_iou_change.py',
	'../models/cascade_rcnn_r50_fpn-064.py',
	'../datasets/coco_detection_custom_Bright_Satura.py', #0.61 base
	'../default_runtime.py',
	'../schedules/schedule_3x_cosinerestart.py'
]

# Load pretrained Swin-S model
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'

swin_b='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'

# set model backbone to Swin-S
model = dict(
	backbone=dict(
		_delete_=True,
		type='SwinTransformer',
		embed_dims=128,
		depths=[2, 2, 18, 2],
		num_heads=[4, 8, 16, 32],
		window_size=7,
		mlp_ratio=4,
		qkv_bias=True,
		qk_scale=None,
		drop_rate=0.,
		attn_drop_rate=0.,
		drop_path_rate=0.2,
		patch_norm=True,
		out_indices=(0, 1, 2, 3),
		with_cp=False,
		convert_weights=True,
		init_cfg=dict(type='Pretrained', checkpoint=swin_b)),
	neck=dict(in_channels=[128, 256, 512, 1024])
)

# Mixed Precision training
#fp16 = dict(loss_scale=512.)
