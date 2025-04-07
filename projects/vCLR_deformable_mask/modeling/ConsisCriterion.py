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
import torch.nn as nn


class ConsisCriterion(nn.Module):
    def __init__(
        self,
        matcher,
        weight_dict,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        
        
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, siamese_outputs, targets, dn_meta):
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
        
        for kk in range(len(indices)):
            sorted_y, indices_y = torch.sort(indices[kk][1])
            sorted_x = indices[kk][0][indices_y]
            indices[kk] = (sorted_x, sorted_y)
            
        
        indices_siamese = self.matcher(siamese_outputs, targets)
        
        for kk in range(len(indices_siamese)):
            sorted_y, indices_y = torch.sort(indices_siamese[kk][1])
            sorted_x = indices_siamese[kk][0][indices_y]
            indices_siamese[kk] = (sorted_x, sorted_y)    
        
        
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
        losses.update(self.cosineSimilarityLoss(outputs, siamese_outputs, indices, indices_siamese, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # if "aux_outputs" in outputs:
        #     for i, aux_outputs in enumerate(outputs["aux_outputs"]):
        #         indices = self.matcher(aux_outputs, targets)
        #         l_dict = self.cosineSimilarityLoss(aux_outputs, targets, indices, num_boxes)
        #         l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
        #         losses.update(l_dict)

        return losses
    
    def cosineSimilarityLoss(self, outputs, siamese_outputs, indices, indices_siamese, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_features = outputs["pred_queries"][idx]
        
        target_idx = self._get_src_permutation_idx(indices_siamese)
        target_features = siamese_outputs["pred_query"][target_idx]

        loss_sim = -nn.functional.cosine_similarity(src_features, target_features.detach(), dim=1).mean()
    
        losses = {}
        losses["loss_sim"] = loss_sim
        return losses 
        
