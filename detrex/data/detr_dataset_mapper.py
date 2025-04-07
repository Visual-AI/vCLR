# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/d2/detr/dataset_mapper.py
# ------------------------------------------------------------------------------------------------

import copy
import logging
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from pycocotools import mask as coco_mask
import cv2
import random

__all__ = ["DetrDatasetMapper", "CustominsDetrDatasetMapper", "CustomMixedDetrDatasetMapper"]


class DetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into the format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors

    Args:
        augmentation (list[detectron.data.Transforms]): The geometric transforms for
            the input raw image and annotations.
        augmentation_with_crop (list[detectron.data.Transforms]): The geometric transforms with crop.
        is_train (bool): Whether to load train set or val set. Default: True.
        mask_on (bool): Whether to return the mask annotations. Default: False.
        img_format (str): The format of the input raw images. Default: RGB.

    Because detectron2 did not implement `RandomSelect` augmentation. So we provide both `augmentation` and
    `augmentation_with_crop` here and randomly apply one of them to the input raw images.
    """

    def __init__(
        self,
        augmentation,
        augmentation_with_crop,
        is_train=True,
        mask_on=False,
        img_format="RGB",
        augmentation_strong=None,
    ):
        self.mask_on = mask_on
        self.augmentation = augmentation
        self.augmentation_with_crop = augmentation_with_crop
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(
                str(self.augmentation), str(self.augmentation_with_crop)
            )
        )

        self.img_format = img_format
        self.is_train = is_train
        self.augmentation_strong = augmentation_strong

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        
        # modified by cbzhang
        file_path = dataset_dict["file_name"]
        rgb_file_path = file_path
        if 'train2017' in file_path:
            r = random.random() * 3
            if r < 1:
                file_path = file_path.replace("train2017", "style_coco_train2017")
                file_path = file_path.replace("datasets", "/home/cbzhang")
            elif r > 2:
                file_path = file_path.replace("train2017", "train2017_depth_cmap")
                file_path = file_path.replace(".jpg", ".png")
                file_path = file_path.replace("datasets", "/home/cbzhang")
            
            r = random.random() * 3
            if r < 1:
                rgb_file_path = rgb_file_path.replace("train2017", "style_coco_train2017")
                rgb_file_path = rgb_file_path.replace("datasets", "/home/cbzhang")
            elif r > 2:
                rgb_file_path = rgb_file_path.replace("train2017", "train2017_depth_cmap")
                rgb_file_path = rgb_file_path.replace(".jpg", ".png")
                rgb_file_path = rgb_file_path.replace("datasets", "/home/cbzhang")
        
        # if 'train' in file_path:
        #     r = random.random() * 3
        #     if r < 1: 
        #         file_path = file_path.replace("CLEVR_v1.0/images/train/", "stylized_clevr_training/").replace(".png", ".jpg")
        #     elif r > 2:
        #         file_path = file_path.replace("CLEVR_v1.0/images/train/", "CLEVR_TRAIN_DEPTHMAP/")
        
        image = utils.read_image(file_path, format=self.img_format)
        
        
        
        # depthmap_path = 'datasets/' + dataset_dict['file_name'].split('/')[2] + '_depthmap/' + dataset_dict['file_name'].split('/')[-1][:-4]+'.png'
        # depthmap = utils.read_image(depthmap_path, format='L')
        # image = np.concatenate([image, depthmap], axis=2)
        
        utils.check_image_size(dataset_dict, image)
        if self.augmentation_with_crop is None:
            image, transforms = T.apply_transform_gens(self.augmentation, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.augmentation, image)
            else:
                image, transforms = T.apply_transform_gens(self.augmentation_with_crop, image)
        
        # cbzhang
        image_rgb = utils.read_image(rgb_file_path, format=self.img_format)
        image_rgb = transforms.apply_image(image_rgb)

        image_shape = image.shape[:2]  # h, w
        
        # strong augmentation version
        if self.augmentation_strong is not None:
            image_strong, _ = T.apply_transform_gens(self.augmentation_strong, image.copy())

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["image_rgb"] = torch.as_tensor(np.ascontiguousarray(image_rgb.transpose(2, 0, 1)))
        if self.augmentation_strong is not None:
            dataset_dict["image_strong"] = torch.as_tensor(np.ascontiguousarray(image_strong.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)


            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            
            # add a foreground mask
            # dataset_dict["fore_mask"] = convert_coco_poly_to_mask(dataset_dict['instances'].gt_masks, dataset_dict["image"].shape[1], dataset_dict["image"].shape[2])
            # dataset_dict["fore_mask"] = dataset_dict["fore_mask"].sum(0)
            # dataset_dict["fore_mask"][dataset_dict["fore_mask"] > 0] = 1
        return dataset_dict


# def convert_coco_poly_to_mask(segmentations, height, width):
#     masks = []
#     for polygons in segmentations:
#         rles = coco_mask.frPyObjects(polygons, height, width)
#         mask = coco_mask.decode(rles)
#         if len(mask.shape) < 3:
#             mask = mask[..., None]
#         mask = torch.as_tensor(mask, dtype=torch.uint8)
#         mask = mask.any(dim=2)
#         masks.append(mask)
#     if masks:
#         masks = torch.stack(masks, dim=0)
#     else:
#         masks = torch.zeros((0, height, width), dtype=torch.uint8)
#     return masks





# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/d2/detr/dataset_mapper.py
# ------------------------------------------------------------------------------------------------

import copy
import logging
import numpy as np
import torch
from detectron2.structures import Instances, Boxes, PolygonMasks
from pycocotools import mask as coco_mask


from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def convert_coco_rle_to_mask(segmentations, height, width):
    mask = torch.as_tensor(segmentations.tensor, dtype=torch.uint8)
    return mask
    

class CustominsDetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into the format used by DETR.
    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors

    Args:
        augmentation (list[detectron.data.Transforms]): The geometric transforms for
            the input raw image and annotations.
        augmentation_with_crop (list[detectron.data.Transforms]): The geometric transforms with crop.
        is_train (bool): Whether to load train set or val set. Default: True.
        mask_on (bool): Whether to return the mask annotations. Default: False.
        img_format (str): The format of the input raw images. Default: RGB.

    Because detectron2 did not implement `RandomSelect` augmentation. So we provide both `augmentation` and
    `augmentation_with_crop` here and randomly apply one of them to the input raw images.
    """

    def __init__(
        self,
        augmentation,
        augmentation_with_crop,
        is_train=True,
        mask_on=False,
        img_format="RGB",
        augmentation_strong=None,
        instance_mask_format='polygon'
    ):
        self.mask_on = mask_on
        self.augmentation = augmentation
        self.augmentation_with_crop = augmentation_with_crop
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(
                str(self.augmentation), str(self.augmentation_with_crop)
            )
        )

        self.img_format = img_format
        self.is_train = is_train
        self.augmentation_strong = augmentation_strong
        self.instance_mask_format = instance_mask_format

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        padding_mask = np.ones(image.shape[:2])
        if self.augmentation_with_crop is None:
            image, transforms = T.apply_transform_gens(self.augmentation, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.augmentation, image)
            else:
                image, transforms = T.apply_transform_gens(self.augmentation_with_crop, image)

        image_shape = image.shape[:2]  # h, w
        
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format=
                                        self.instance_mask_format)
            
            # if not instances.has('gt_masks'): 
            #     instances.gt_masks = PolygonMasks([])  # for negative examples
            # instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            
            
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                if self.instance_mask_format == 'polygon':
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                else:
                    gt_masks = convert_coco_rle_to_mask(gt_masks, h, w)
                instances.gt_masks = gt_masks
            # import ipdb; ipdb.set_trace()
            dataset_dict["instances"] = instances
        return dataset_dict




class CustomMixedDetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into the format used by DETR.
    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors

    Args:
        augmentation (list[detectron.data.Transforms]): The geometric transforms for
            the input raw image and annotations.
        augmentation_with_crop (list[detectron.data.Transforms]): The geometric transforms with crop.
        is_train (bool): Whether to load train set or val set. Default: True.
        mask_on (bool): Whether to return the mask annotations. Default: False.
        img_format (str): The format of the input raw images. Default: RGB.

    Because detectron2 did not implement `RandomSelect` augmentation. So we provide both `augmentation` and
    `augmentation_with_crop` here and randomly apply one of them to the input raw images.
    """

    def __init__(
        self,
        augmentation,
        augmentation_with_crop,
        is_train=True,
        mask_on=False,
        img_format="RGB",
        augmentation_strong=None,
        instance_mask_format='polygon'
    ):
        self.mask_on = mask_on
        self.augmentation = augmentation
        self.augmentation_with_crop = augmentation_with_crop
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(
                str(self.augmentation), str(self.augmentation_with_crop)
            )
        )

        self.img_format = img_format
        self.is_train = is_train
        self.augmentation_strong = augmentation_strong
        self.instance_mask_format = instance_mask_format

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        padding_mask = np.ones(image.shape[:2])
        if self.augmentation_with_crop is None:
            image, transforms = T.apply_transform_gens(self.augmentation, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.augmentation, image)
            else:
                image, transforms = T.apply_transform_gens(self.augmentation_with_crop, image)

        image_shape = image.shape[:2]  # h, w
        
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format=
                                        self.instance_mask_format)
            
            # if not instances.has('gt_masks'): 
            #     instances.gt_masks = PolygonMasks([])  # for negative examples
            # instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            
            
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                if self.instance_mask_format == 'polygon':
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                else:
                    gt_masks = convert_coco_rle_to_mask(gt_masks, h, w)
                instances.gt_masks = gt_masks
            # import ipdb; ipdb.set_trace()
            dataset_dict["instances"] = instances
        return dataset_dict
