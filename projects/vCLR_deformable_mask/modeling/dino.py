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

import copy
import math
import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers.nms import batched_nms
import pickle

from detrex.modeling import ema

from detectron2.structures import Instances, ROIMasks


# perhaps should rename to "resize_instance"
def detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]
    # if results.has("pred_masks"):
    #     if isinstance(results.pred_masks, ROIMasks):
    #         roi_masks = results.pred_masks
    #     else:
    #         # pred_masks is a tensor of shape (N, 1, M, M)
    #         roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
    #     results.pred_masks = roi_masks.to_bitmasks(
    #         results.pred_boxes, output_height, output_width, mask_threshold
    #     ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results

class Permute(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(0, 2, 1)

class DINO(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],#, 91.2545],
        pixel_std: List[float] = [58.395, 57.120, 57.375],#, 60.0240],
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        input_format: Optional[str] = "RGB",
        vis_period: int = 0,
        depth_net: nn.Module = None,
        r50_extractor: nn.Module = None,
        consistency_criterion: nn.Module = None,
    ):
        super().__init__()
        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = transformer
        
        # define depth model
        self.depth_model = depth_net
        
        self.r50_extractor = r50_extractor
        

        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion
        
        self.consistency_criterion = consistency_criterion

        # denoising
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # initialize weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # two-stage
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # the period for visualizing training samples
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"
        self.training_or_not = None #
        
        self.ema_model_state_dict = None
        self.load_r50_extractor_ckpt = False 
        
        
        self.ROI_embed = nn.Sequential(
            MLP(embed_dim, embed_dim, 1024, 3),
            nn.ReLU()
        )
        self.ROI_embed = nn.ModuleList([copy.deepcopy(self.ROI_embed) for i in range(num_pred)])
        if self.r50_extractor is not None:
            for name, param in self.r50_extractor.named_parameters():
                param.requires_grad = False
        
        # for mask prediction
        self.mask_embed = MLP(embed_dim, embed_dim, 1024, 3)
        self.mask_embed = nn.ModuleList([copy.deepcopy(self.mask_embed) for i in range(num_pred)])

        # two-stage
        self.transformer.decoder.mask_embed = self.mask_embed
        
        self.mapping_fpn_features_for_seg = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, 1, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 1024, 3, 1, 1),
        )
        self.post_layernorm = nn.LayerNorm(1024)
        
        self.add_noise = None
        
    def image_transform(self, batched_inputs, images_transformed):
        images_transformed = self.random_mix(batched_inputs, images_transformed)
        images_transformed = self.random_erase(batched_inputs, images_transformed)
        images_transformed = self.random_grayscale(images_transformed)
        return images_transformed

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        if self.training:
            images_transformed = self.preprocess_image_strong(batched_inputs)
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        
        if self.training:
            images_weak = self.prepare_weak_images(batched_inputs, (H, W))
            with torch.no_grad():
                siamese_results = self.infer_results(batched_inputs, images_weak, img_masks)
        
        if self.training == False:
            return self.forward_student(batched_inputs, images, img_masks)
        else:
            images_transformed = self.image_transform(batched_inputs, images_transformed)
            
            loss_dict = self.forward_student(batched_inputs, images_transformed, img_masks, ema_gts=None, weak_images=images, siamese_outputs=siamese_results)

            return loss_dict
    
    
    def infer_results(self, batched_inputs, images, img_masks):
        with torch.no_grad():
            param_req_grad = {}
            for name, param in self.named_parameters():
                param_req_grad[name] = param.requires_grad
                param.requires_grad = False
            
            # import pdb; pdb.set_trace()
            # r_flip = random.random()
            # if r_flip < 0.5:
            #     images.tensor = torch.flip(images.tensor, dims=(3,))
            with ema.apply_model_ema_and_restore(self):
                features = self.backbone(images.tensor)  # output feature dict
                
                multi_level_feats = self.neck(features)
                multi_level_masks = []
                multi_level_position_embeddings = []
                for feat in multi_level_feats:
                    multi_level_masks.append(
                        F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
                    )
                    multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

                # if r_flip < 0.5:
                #     for kk in range(len(multi_level_feats)):
                #         import pdb; pdb.set_trace()
                #         multi_level_feats = (1 - multi_level_masks[kk]).unsqueeze(1) * multi_level_feats[kk]
                #         _nc = multi_level_feats[kk].shape[1]
                #         _mask = multi_level_masks[kk].unsqueeze(1).repeat(1, _nc, 1, 1)
                #         multi_level_feats[kk][_mask == 0] = torch.flip(multi_level_feats[kk][_mask == 0], dims=(3,))
    
                input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
                query_embeds = (input_query_label, input_query_bbox)

                # feed into transformer
                (
                    inter_states,
                    init_reference,
                    inter_references,
                    enc_state,
                    enc_reference,  # [0..1]
                    memory,
                ) = self.transformer(
                    multi_level_feats,
                    multi_level_masks,
                    multi_level_position_embeddings,
                    query_embeds,
                    attn_masks=[attn_mask, None],
                )
                # hack implementation for distributed training
                inter_states[0] += self.label_enc.weight[0, 0] * 0.0

                # Calculate output coordinates and classes.
                outputs_classes = []
                outputs_coords = []
                outputs_rois = []
                
                outputs_pred_queries = [] # TODO
                
                for lvl in range(inter_states.shape[0]):
                    if lvl == 0:
                        reference = init_reference
                    else:
                        reference = inter_references[lvl - 1]
                    reference = inverse_sigmoid(reference)
                    outputs_class = self.class_embed[lvl](inter_states[lvl])
                    
                    outputs_roi = self.ROI_embed[lvl](inter_states[lvl])
                    
                    tmp = self.bbox_embed[lvl](inter_states[lvl])
                    if reference.shape[-1] == 4:
                        tmp += reference
                    else:
                        assert reference.shape[-1] == 2
                        tmp[..., :2] += reference
                    outputs_coord = tmp.sigmoid()
                    
                    # outputs_pred_query = self.predictor[lvl](inter_states[lvl])
                    outputs_pred_query = inter_states[lvl].clone().detach()
                    
                    outputs_classes.append(outputs_class)
                    outputs_coords.append(outputs_coord)
                    outputs_rois.append(outputs_roi)
                    outputs_pred_queries.append(outputs_pred_query)
                    
                outputs_class = torch.stack(outputs_classes)
                # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
                outputs_coord = torch.stack(outputs_coords)
                # tensor shape: [num_decoder_layers, bs, num_query, 4]
                outputs_roi = torch.stack(outputs_rois)

                outputs_pred_query = torch.stack(outputs_pred_queries) # TODO

                # denoising postprocessing
                if dn_meta is not None:
                    outputs_class, outputs_coord, outputs_roi, outputs_mask = self.dn_post_process(
                        outputs_class, outputs_coord, dn_meta, outputs_roi
                    )
                # prepare for loss computation
                output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1], "pred_rois": outputs_roi[-1], "pred_query": outputs_pred_query[-1]}                
                    
                # prepare two stage output
                interm_coord = enc_reference
                interm_class = self.transformer.decoder.class_embed[-1](enc_state)
                interm_roi = self.ROI_embed[-1](enc_state)
                output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord, "pred_rois": interm_roi, "pred_query": enc_state}

                for name, param in self.named_parameters():
                    param.requires_grad = param_req_grad[name]
                return output
                
                # without nms to save training time
                
                # box_cls = output["pred_logits"]
                # box_pred = output["pred_boxes"]
                # box_pred_query = output["pred_queries"]
                # results = self.nms_inference(box_cls, box_pred, images.image_sizes, topk=100, pred_query=box_pred_query) # NMS
                                
                # enc_results = self.nms_inference(output["enc_outputs"]["pred_logits"], output["enc_outputs"]["pred_boxes"], images.image_sizes, topk=100, pred_query=output["enc_outputs"]["pred_query"]) # NMS
                # processed_results_enc = []
                # for results_per_image, input_per_image, image_size in zip(
                #     enc_results, batched_inputs, images.image_sizes
                # ):
                #     height = input_per_image.get("height", image_size[0])
                #     width = input_per_image.get("width", image_size[1])
                #     results_per_image.pred_boxes.scale(scale_x=1./image_size[1], scale_y=1./image_size[0])                
                #     processed_results_enc.append({
                #         "boxes": box_xyxy_to_cxcywh(results_per_image.pred_boxes.tensor),
                #         "labels": results_per_image.pred_classes,
                #         "pred_query": results_per_image.pred_query
                #     })
                
                # # # process queries for negative exampels
                # # processed_results_negative = []
                # # for results_per_image, input_per_image, image_size in zip(
                # #     negative_results, batched_inputs, images.image_sizes
                # # ):
                # #     height = input_per_image.get("height", image_size[0])
                # #     width = input_per_image.get("width", image_size[1])
                # #     results_per_image.pred_boxes.scale(scale_x=1./image_size[1], scale_y=1./image_size[0])                
                # #     processed_results_negative.append({
                # #         "boxes": box_xyxy_to_cxcywh(results_per_image.pred_boxes.tensor),
                # #         "labels": results_per_image.pred_classes,
                # #         "pred_query": results_per_image.pred_query
                # #     })
                
                # processed_results = []
                # for results_per_image, input_per_image, image_size in zip(
                #     results, batched_inputs, images.image_sizes
                # ):
                #     height = input_per_image.get("height", image_size[0])
                #     width = input_per_image.get("width", image_size[1])
                #     results_per_image.pred_boxes.scale(scale_x=1./image_size[1], scale_y=1./image_size[0])                
                #     processed_results.append({
                #         "boxes": box_xyxy_to_cxcywh(results_per_image.pred_boxes.tensor),
                #         "labels": results_per_image.pred_classes,
                #         "pred_query": results_per_image.pred_query
                #     })
            
            for name, param in self.named_parameters():
                param.requires_grad = param_req_grad[name]
                
            return processed_results, processed_results_enc
    
    
    
    def prepare_weak_images(self, batched_inputs, padding_size):
        images_weak = []
        for x in batched_inputs: 
            img = self.normalizer(x['image_rgb'].to(self.device)) # 3 870 736
            pad_right = padding_size[1] - img.shape[2]
            pad_bottom = padding_size[0] - img.shape[1]
            img = torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom), value=0.)
            images_weak.append(img)
        images_weak = ImageList.from_tensors(images_weak)        
        return images_weak
    
    # transform
    def random_mix(self, batched_inputs, images_strong):
        '''images: [b c h w] after normalized
        '''
        results = []
        batch_size, _, H, W = images_strong.tensor.shape
        for img_id in range(batch_size):
            img_h, img_w = batched_inputs[img_id]["instances"].image_size
            # # perform random erase
            # n_pathes = np.random.randint(0, 60) # 16x16 
            # h_axis, w_axis = torch.randperm(img_h), torch.randperm(img_w)
            
            # perform random mixip using background
            box_h, box_w = img_h // 8, img_w // 8
            x0 = torch.randint(low=0, high=img_w-box_w, size=(1,))
            y0 = torch.randint(low=0, high=img_h-box_h, size=(1,))
            img_patch = images_strong.tensor[img_id:img_id+1, :, y0:y0+box_h, x0:x0+box_w]
            img_patch = torch.nn.functional.interpolate(img_patch, (img_h, img_w), mode='bilinear')
            ratio = torch.abs(torch.rand(1).to(images_strong.tensor.device) - 0.5) + 0.5
            # ratio = torch.rand(1).to(images_strong.tensor.device)
            images_strong.tensor[img_id, :, :img_h, :img_w] = images_strong.tensor[img_id, :, :img_h, :img_w] * ratio + img_patch[0] * (1. - ratio)
        
        return images_strong
    
    # augmentation
    def random_mix_images(self, batched_inputs, images_strong, targets, weak_images):
        batch_size, _, H, W = images_strong.tensor.shape
        for img_id in range(batch_size - 1):
            img_h, img_w = batched_inputs[img_id]["instances"].image_size
            mix_img_id = (img_id + 1) % batch_size
            img_h_mix, img_w_mix = batched_inputs[mix_img_id]["instances"].image_size
            mix_image = torch.nn.functional.interpolate(weak_images.tensor[mix_img_id:mix_img_id+1, :, :img_h_mix, :img_w_mix], (img_h, img_w), mode='bilinear', align_corners=True)
            
            mix_ratio = random.random()
            images_strong.tensor[img_id, :, :img_h, :img_w] = images_strong.tensor[img_id, :, :img_h, :img_w] * mix_ratio + \
                mix_image * (1 - mix_ratio)

            targets[img_id]['labels'] = torch.cat([targets[img_id]['labels'], targets[mix_img_id]['labels']], dim=0)
            targets[img_id]['boxes'] = torch.cat([targets[img_id]['boxes'], targets[mix_img_id]['boxes']], dim=0)            
        
        return images_strong, targets
    
    def random_grayscale(self, images_strong):
        if random.random() > 0.5:
            pixel_mean = torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1).to(images_strong.tensor.device)
            pixel_std = torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1).to(images_strong.tensor.device)
            images_strong.tensor = images_strong.tensor * pixel_std + pixel_mean
            images_strong.tensor = (0.299 * images_strong.tensor[:, 0] + 0.587 * images_strong.tensor[:, 1] + \
                images_strong.tensor[:, 2] * 0.114).unsqueeze(1).repeat(1, 3, 1, 1)
            images_strong.tensor = (images_strong.tensor - pixel_mean) / pixel_std
        return images_strong
    
    # # augmentation
    def random_erase(self, batched_inputs, images_strong):
        '''images: [b c h w] after normalized
        '''
        results = []
        min_area = 0.02
        max_area = 1. / 3.
        min_aspect = 0.3
        max_aspect = 0.7
        batch_size, _, H, W = images_strong.tensor.shape
        log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        for img_id in range(batch_size):
            img_h, img_w = batched_inputs[img_id]["instances"].image_size
            target_area = random.uniform(min_area, max_area) * img_h * img_w
            aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if h < img_h and w < img_w:
                pass
            else:
                h = int(img_h / 2.)
                w = int(img_w / 2.)
            top = random.randint(0, img_h - h)
            left = random.randint(0, img_w - w)
            images_strong.tensor[img_id, :, top:top+h, left:left+w] = 0.
        
        return images_strong
        
            
            
            
    def forward_ema(self, batched_inputs, images, img_masks):
        with torch.no_grad():
            if self.training_or_not is None:
                self.training_or_not = {}
                for i, (name, params) in enumerate(self.named_parameters()):
                    self.training_or_not[name] = params.requires_grad
                        
            curr_state = copy.deepcopy(self.state_dict())
            # ema_state = copy.deepcopy(self.ema_state.state_dict())
            ema_state = self.ema_model_state_dict
            # for k in curr_state.keys():
            #     if k not in ema_state.keys():
            #         ema_state[k] = ema_state['transformer.decoder.' + k]
            
            self.load_state_dict(ema_state, strict=False)
            for i, (name, params) in enumerate(self.named_parameters()):
                params.requires_grad = False
                        
            # self.load_state_dict(ema_state, strict=False)
            # self.eval()
            
            features = self.backbone(images.tensor)  # output feature dict

            # project backbone features to the reuired dimension of transformer
            # we use multi-scale features in DINO
            multi_level_feats = self.neck(features)
            multi_level_masks = []
            multi_level_position_embeddings = []
            for feat in multi_level_feats:
                multi_level_masks.append(
                    F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
                )
                multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
            query_embeds = (input_query_label, input_query_bbox)
            # feed into transformer
            (
                inter_states,
                init_reference,
                inter_references,
                enc_state,
                enc_reference,  # [0..1]
            ) = self.transformer(
                multi_level_feats,
                multi_level_masks,
                multi_level_position_embeddings,
                query_embeds,
                attn_masks=[attn_mask, None],
            )
            # hack implementation for distributed training
            inter_states[0] += self.label_enc.weight[0, 0] * 0.0

            # Calculate output coordinates and classes.
            outputs_classes = []
            outputs_coords = []
            outputs_rois = []
            for lvl in range(inter_states.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.class_embed[lvl](inter_states[lvl])
                
                outputs_roi = self.ROI_embed[lvl](inter_states[lvl])
                
                tmp = self.bbox_embed[lvl](inter_states[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
                outputs_rois.append(outputs_roi)
                
            outputs_class = torch.stack(outputs_classes)
            # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
            outputs_coord = torch.stack(outputs_coords)
            # tensor shape: [num_decoder_layers, bs, num_query, 4]
            outputs_roi = torch.stack(outputs_rois)

            # prepare for loss computation
            output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1], "pred_rois": outputs_roi[-1]}
            if self.aux_loss:
                output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

            # prepare two stage output
            interm_coord = enc_reference
            interm_class = self.transformer.decoder.class_embed[-1](enc_state)
            output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

        
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            # make pseudo labels
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            
            results = []
            bs, n_queries, n_cls = box_cls.shape

            # Select top-k confidence boxes for inference
            prob = box_cls.sigmoid()

            all_scores = prob.view(bs, n_queries * n_cls).to(box_cls.device)
            all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(box_cls.device)
            all_boxes = torch.div(all_indexes, box_cls.shape[2], rounding_mode="floor")
            all_labels = all_indexes % box_cls.shape[2]

            # convert to xyxy for nms post-process
            boxes = box_cxcywh_to_xyxy(box_pred)
            boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1, 1, 4))

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
                zip(all_scores, all_labels, boxes, images.image_sizes)
            ):

                pre_topk = scores_per_image.topk(900).indices
                box = box_pred_per_image[pre_topk]
                score = scores_per_image[pre_topk]
                label = labels_per_image[pre_topk]
                # filter by confidence
                
                # concat gt boxes
                target_boxes = box_cxcywh_to_xyxy(targets[i]['boxes'])
                box = torch.cat([box, target_boxes], dim=0) # N 4
                label = torch.cat([label, targets[i]['labels']]) # N
                score = torch.cat([score, targets[i]['labels'] + 1]) # N 
                #filter
                
                box = box[(score > 0.3)]
                label = label[(score > 0.3)]
                score = score[(score > 0.3)]

                # nms post-process
                keep_index = batched_nms(box, score, label, 0.5)
                
                keep_index = keep_index[score[keep_index] < 1.]
                
                # num_gts = min(keep_index.shape[0], targets[i]['labels'].shape[0])
                # keep_index = keep_index[:min(num_gts + 2, keep_index.shape[0])]
                keep_index = keep_index[:min(5, keep_index.shape[0])]

                result = dict(labels=label[keep_index].detach(), boxes=box_xyxy_to_cxcywh(box[keep_index]).detach())
                
                results.append(result)
            
            self.load_state_dict(curr_state)
            for i, (name, params) in enumerate(self.named_parameters()):
                params.requires_grad = self.training_or_not[name]
                
            # curr_state.apply_to(ema._remove_ddp(self)) # restore model state
            # self.train()
                
            return results
        
    
    def forward_student(self, batched_inputs, images, img_masks, ema_gts=None, weak_images=None, siamese_outputs=None):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images)

        

        # original features
        features = self.backbone(images.tensor)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        
        
        # denoising preprocessing
        # prepare label query embedding
        if self.training:
            # add ema_gts in targets TODO
            if ema_gts is not None:
                for bs in range(len(ema_gts)):
                    if ema_gts[bs]['labels'].shape[0] == 0:
                        continue
                    targets[bs]['labels'] = torch.cat([targets[bs]['labels'], ema_gts[bs]['labels']], dim=0)
                    targets[bs]['boxes'] = torch.cat([targets[bs]['boxes'], ema_gts[bs]['boxes']])
            
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.embed_dim,
                label_enc=self.label_enc,
            )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
            enc_memory # fpn output of encoder
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
        )
        # hack implementation for distributed training
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0

        fpn_features_seg = []
        resize_h, resize_w = multi_level_feats[0].shape[2:]
        start_idx = 0
        for feat in multi_level_feats:
            hh, ww = feat.shape[-2:]
            fpn_features_seg.append(torch.nn.functional.interpolate(enc_memory[:, start_idx:start_idx+hh*ww, :].reshape(enc_memory.shape[0], hh, ww, enc_memory.shape[2]).permute(0, 3, 1, 2),
                            (resize_h, resize_w), mode='bilinear', align_corners=True))
            start_idx += hh * ww
        
        fpn_features_seg = torch.cat(fpn_features_seg, dim=1)
        fpn_features_seg = self.mapping_fpn_features_for_seg(fpn_features_seg) + fpn_features_seg # add params
        fpn_features_seg = self.post_layernorm(fpn_features_seg.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        
        
        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        outputs_rois = []
        
        outputs_pred_queries = [] # TODO
        
        outputs_pred_masks = []
        
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            
            outputs_roi = self.ROI_embed[lvl](inter_states[lvl])
            outputs_masks = self.mask_embed[lvl](inter_states[lvl]) # B 2600 512
            
            outputs_masks = torch.bmm(outputs_masks, fpn_features_seg.flatten(2)).reshape(outputs_masks.shape[0], outputs_masks.shape[1], resize_h, resize_w)
            outputs_masks = outputs_masks
            
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            
            # outputs_pred_query = self.predictor[lvl](inter_states[lvl])
            
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_rois.append(outputs_roi)
            
            outputs_pred_queries.append(inter_states[lvl])
            outputs_pred_masks.append(outputs_masks) # after sigmoid
            
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]
        outputs_roi = torch.stack(outputs_rois)

        outputs_pred_query = torch.stack(outputs_pred_queries) # TODO
        outputs_mask = torch.stack(outputs_pred_masks)
        
        # denoising postprocessing
        if dn_meta is not None:
            outputs_class, outputs_coord, outputs_roi, outputs_mask = self.dn_post_process(
                outputs_class, outputs_coord, dn_meta, outputs_roi, outputs_pred_query, outputs_mask
            )
            
            # TODO cbzhang
            if dn_meta['single_padding'] > 0:
                padding_size = dn_meta["single_padding"] * dn_meta["dn_num"]
                outputs_pred_query = outputs_pred_query[:, :, padding_size:, :]
            
            
            
        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1], "pred_rois": outputs_roi[-1], \
            "pred_queries": outputs_pred_query[-1], "pred_masks": outputs_mask[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord, outputs_roi, outputs_pred_query, outputs_mask)
            
            
        # prepare two stage output
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        interm_roi = self.ROI_embed[-1](enc_state)
        
        outputs_masks_enc = self.mask_embed[-1](enc_state) # B 2600 512
        outputs_masks_enc = torch.bmm(outputs_masks_enc, fpn_features_seg.flatten(2)).reshape(outputs_masks_enc.shape[0], outputs_masks_enc.shape[1], resize_h, resize_w)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord, "pred_rois": interm_roi, "pred_masks": outputs_masks_enc}

        if self.training:
            # compute loss
            loss_dict = self.criterion(output, targets, dn_meta)
            if siamese_outputs is not None and self.consistency_criterion is not None:
                loss_consistency = self.consistency_criterion(output, siamese_outputs, targets, dn_meta)
                loss_dict.update(loss_consistency)
            
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"] 
            mask_pred = output["pred_masks"] # cbzhang
            
            mask_score = ((mask_pred > 0) * mask_pred.sigmoid()).sum((2,3)) / ((mask_pred > 0).sum((2, 3))+1e-10)
            avg_score = inverse_sigmoid(torch.sqrt(box_cls.sigmoid() * mask_score.unsqueeze(-1)))
            
            results = self.nms_inference(avg_score, box_pred, mask_pred, images.image_sizes) # NMS
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                pred_mask = torch.nn.functional.interpolate(
                    results_per_image.pred_masks.unsqueeze(1),
                    (height, width),
                    # (10, 10),
                    mode='bilinear', align_corners=False)
                pred_mask = (pred_mask[:, 0] > 0).float()
                r = detector_postprocess(results_per_image, height, width)
                r.pred_masks = pred_mask 
                processed_results.append({"instances": r})
            return processed_results

    def visualize_training(self, batched_inputs, results):
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_box = 20

        for input, results_per_image in zip(batched_inputs, results):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=results_per_image.pred_boxes[:max_vis_box].tensor.detach().cpu().numpy()
            )
            pred_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, pred_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted boxes"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_roi, outputs_query, outputs_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_rois": c, "pred_queries": d, "pred_masks": e}
            for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1], outputs_roi[:-1], outputs_query[:-1], outputs_mask[:-1])
        ]

    def prepare_for_cdn(
        self,
        targets,
        dn_number,
        label_noise_ratio,
        box_noise_scale,
        num_queries,
        num_classes,
        hidden_dim,
        label_enc,
    ):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
            # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1
            )  # half of bbox prob
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))

        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        )
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            )
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to("cuda")
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * i * 2
                ] = True
            else:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True

        dn_meta = {
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
        }

        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas, outputs_roi, outputs_pred_query, outputs_pred_mask=None):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            output_known_roi = outputs_roi[:, :, :padding_size, :]
            output_known_query = outputs_pred_query[:, :, :padding_size, :]
            
            output_known_mask = outputs_pred_mask[:, :, :padding_size, :]
            
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]
            outputs_roi = outputs_roi[:, :, padding_size:, :]
            outputs_query = outputs_pred_query[:, :, padding_size:, :]
            outputs_mask = outputs_pred_mask[:, :, padding_size:, :]

            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1],\
                "pred_rois": output_known_roi[-1], "outputs_query": output_known_query[-1], "pred_masks": output_known_mask[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord, output_known_roi, output_known_query, output_known_mask)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord, outputs_roi, outputs_mask

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images
    
    def preprocess_image_strong(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images
    
    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results
    
    def nms_inference(self, box_cls, box_pred, mask_pred, image_sizes, topk=300, pred_query=None):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        bs, n_queries, n_cls = box_cls.shape

        # Select top-k confidence boxes for inference
        prob = box_cls.sigmoid()

        all_scores = prob.view(bs, n_queries * n_cls).to(box_cls.device)
        all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(box_cls.device)
        all_boxes = torch.div(all_indexes, box_cls.shape[2], rounding_mode="floor")
        all_labels = all_indexes % box_cls.shape[2]

        # convert to xyxy for nms post-process
        boxes = box_cxcywh_to_xyxy(box_pred)
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1, 1, 4))
        
        for i, (scores_per_image, labels_per_image, box_pred_per_image, mask_pred_per_image, image_size) in enumerate(
            zip(all_scores, all_labels, boxes, mask_pred, image_sizes)
        ):

            pre_topk = scores_per_image.topk(topk).indices
            box = box_pred_per_image[pre_topk]
            score = scores_per_image[pre_topk]
            label = labels_per_image[pre_topk]
            mask = mask_pred_per_image[pre_topk]

            # nms post-process
            keep_index = batched_nms(box, score, label, 0.7)

            result = Instances(image_size)
            result.pred_boxes = Boxes(box[keep_index])
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = score[keep_index]
            result.pred_classes = label[keep_index]
            result.pred_masks = mask[keep_index]
            if pred_query is not None:
                result.pred_query = pred_query[i][keep_index]
            results.append(result)
        return results

    def prepare_targets(self, targets, images):
        new_targets = []
        h_pad, w_pad = images.tensor.shape[-2:]
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            
            # prepare masks
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "masks": padded_masks})
        return new_targets







