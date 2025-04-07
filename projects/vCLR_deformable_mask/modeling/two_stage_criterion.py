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

import torch

from detrex.modeling.criterion import SetCriterion
from detrex.utils import get_world_size, is_dist_avail_and_initialized
import torch.nn.functional as F
from detrex.layers.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from detrex.modeling.criterion.criterion import sigmoid_focal_loss

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

# from maskdino.maskformer_model import sigmoid_focal_loss
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_boxes

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))



class TwoStageCriterion(SetCriterion):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes", "masks"],
        eos_coef=None,
        loss_class_type="focal_loss",
        alpha: float = 0.25,
        gamma: float = 2,
        two_stage_binary_cls=False,
    ):
        super().__init__(
            num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma
        )
        self.two_stage_binary_cls = two_stage_binary_cls
        self.num_points = 12544
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # for two stage
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            if self.two_stage_binary_cls:
                for bt in targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # Computation classification loss
        if self.loss_class_type == "ce_loss":
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
        elif self.loss_class_type == "focal_loss":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_class = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * src_logits.shape[1]
            )
        elif self.loss_class_type == 'vfl_loss':
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
            src_logits = outputs['pred_logits']
            target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

            target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
            target_score_o[idx] = ious.to(target_score_o.dtype)
            target_score = target_score_o.unsqueeze(-1) * target
            
            pred_score = torch.sigmoid(src_logits) # NOTE
            
            # target_score = torch.sqrt(target_score) * target + torch.pow(pred_score, 2) * (1 - target)
            # weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score * target           
            # loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
            
            
            # target_score = torch.sqrt(target_score) * target + torch.sqrt(1 - pred_score) * (1 - target)
            pred_score = pred_score * target + (1 - pred_score) * (1 - target)
            pred_score = torch.clamp(pred_score, 1e-7, 1-1e-7)
            target_score = target + torch.sqrt(pred_score) * (1 - target)
            target_score = target_score.detach()
            weight = target_score * (1 - target) + 0.25 * (1 - pred_score.detach())**2 * target
            loss = target_score * torch.log(pred_score + 1e-7) + (1 - target_score) * torch.log(1 - pred_score + 1e-7)
            loss = -loss * weight 
            
            loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
            loss_class = loss
            
        elif self.loss_class_type == 'active_loss':
            batchsize = len(targets)      
            target_score_o = torch.zeros(batchsize, outputs['pred_logits'].shape[1], self.num_classes).to(outputs['pred_logits'].device) # 80 is number of class      
            for bs in range(batchsize):
                src_boxes = outputs['pred_boxes'][bs]
                target_boxes = [targets[bs]['boxes'][i] for i in indices[bs][1]]
                if len(target_boxes) >= 1:
                    target_boxes = torch.stack(target_boxes, dim=0)
                    target_labels = torch.stack([targets[bs]['labels'][i] for i in indices[bs][1]], dim=0)
                    ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)) # 900, gts
                    ious = ious.detach()
                    
                    ious = ious.unsqueeze(2).repeat(1, 1, self.num_classes) # 900, gts, 80
                    target_onehot = F.one_hot(target_labels, num_classes=self.num_classes)
                    target_onehot = ious * target_onehot.unsqueeze(0)  # 900 gts 80
                    target_score_o[bs, :, :] = target_onehot.max(1)[0] # 900 80

            # make new scores for potential objects, which should be (1 - iou_gts) * pred_score
            pred_score = torch.sigmoid(src_logits).detach() 
            target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
            
            novel_object_score = ((1 - target_score_o) >= 0.97) * pred_score * (1 - target)
            novel_scores, novel_idx = torch.topk(novel_object_score, k=20, dim=1)
            # for bs in range(batchsize):
            #     target[bs, novel_idx[bs].squeeze(1)] = (novel_scores[bs] >= 0.5).long()
            #     # if novel_scores[bs][0][0] >= 0.7:
            #     #     target[bs, novel_idx[bs][0][0], :] = 1            
            
            # extract positive iou target
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()            
            target_score_o_pos = torch.zeros_like(target_classes, dtype=src_logits.dtype)
            target_score_o_pos[idx] = ious.to(target_score_o_pos.dtype)
            target_score = target_score_o_pos.unsqueeze(-1) * target 
            target_score = torch.sqrt(target_score)
            
            for bs in range(batchsize):
                target_score[bs, novel_idx[bs].squeeze(1)] = ((novel_scores[bs] >= 0.7) * 0.95).type_as(target_score)
                # if novel_scores[bs][0][0] >= 0.7:
                #     target_score[bs, novel_idx[bs][0][0], :] = 0.9 # NOTE novel box learn 1    
            
            weight = 0.25 * pred_score**2 * (1 - target) + target_score * target 
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
            
            loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
            loss_class = loss
                
        losses = {"loss_class": loss_class}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        
        
        if "pred_rois" in list(outputs.keys()):
            # src_roi = torch.nn.functional.normalize(outputs["pred_rois"][idx])
            # target_roi = torch.nn.functional.normalize(torch.cat([t["features"][i] for t, (_, i) in zip(targets, indices)], dim=0))
            # loss_roi = F.mse_loss(src_roi, target_roi, reduction='none')
            losses["loss_roi"] = outputs["pred_rois"].mean() * 0.
        else:
            losses["loss_roi"] = torch.zeros(1).to(loss_bbox.device)
            
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses
        
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "class": self.loss_labels,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)