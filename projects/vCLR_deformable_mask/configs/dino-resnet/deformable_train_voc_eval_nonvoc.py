from detrex.config import get_config
from ..models.dino_r50 import model

import itertools

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluatorCustom

from detrex.data import CustominsDetrDatasetMapper
from detectron2.modeling.backbone import ResNet, BasicStem
from detrex.data.transforms import ColorAugSSDTransform

from projects.vCLR_deformable_mask.modeling import OursDatasetMapper

dataloader = OmegaConf.create()

# training
register_coco_instances("openworld_voc_classes_train2017", {}, 'datasets/vCLR_voc_train2017_top10.json', 'datasets/coco/train2017/')
register_coco_instances("openworld_nonvoc_classes_val2017", {}, 'datasets/openworld_instances_nonvoc_val2017_insseg.json', 'datasets/coco/val2017/')
register_coco_instances("openworld_uvo_nonvoc_val2017", {}, 'datasets/uvo_nonvoc_val_rle.json', 'datasets/uvo_videos_dense_frames/')

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="openworld_voc_classes_train2017", filter_empty=True), # must be true
    mapper=L(OursDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ], 
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
            # L(T.Resize)(
            #     shape=(1024, 1024)
            # )
        ],
        augmentation_strong=[
            T.RandomApply(T.RandomBrightness(intensity_min=0.5, intensity_max=1.5)),
            T.RandomApply(T.RandomContrast(intensity_min=0.5, intensity_max=1.5)),
            T.RandomApply(T.RandomSaturation(intensity_min=0.5, intensity_max=1.5)),
    
        ],
        # instance_mask_format='bitmask', # for RLE
        is_train=True,
        mask_on=True, # whether return mask
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=16,
)
 
dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="openworld_nonvoc_classes_val2017", filter_empty=True),
    mapper=L(CustominsDetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
            # L(T.Resize)(
            #     shape=(1024, 1024)
            # )
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=True,
        # instance_mask_format='bitmask',
        img_format="RGB", 
    ),
    num_workers=16,
)

dataloader.evaluator = L(COCOEvaluatorCustom)(
    dataset_name="${..test.dataset.names}", max_dets_per_image=[1, 10, 20, 30, 50, 100, 300, 900]
)



# get default config
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_5ep
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/dino_openworld"

# max training iterations
train.max_iter = 60000
train.eval_period = 5000
train.log_period = 200
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4 # original 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 24

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir



# modify model config
model.dn_number = 100
model.num_classes = 1
model.select_box_nums_for_evaluation=900

# ema
train.model_ema.enabled=True
train.model_ema.decay=0.999

model.num_queries = 2000

model.transformer.encoder.use_checkpoint=False 
model.transformer.decoder.use_checkpoint=False
