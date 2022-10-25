# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from torch.nn.init import normal_


from .builder import ROTATED_TRANSFORMER
from mmdet.models.utils import Transformer
from mmdet.models.utils.transformer import inverse_sigmoid
# from mmrotate.core import obb2poly, poly2obb
# # from mmrotate.core import obb2xyxy
# from mmdet.core import bbox_cxcywh_to_xyxy

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
torch.autograd.set_detect_anomaly(True)


def obb2poly_tr(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rboxes[..., 0]
    y = rboxes[..., 1]
    w = rboxes[..., 2]
    h = rboxes[..., 3]
    a = rboxes[..., 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=2)

def bbox_cxcywh_to_xyxy_tr(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy - 0.5 * h),
                (cx - 0.5 * w), (cy + 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)

@ROTATED_TRANSFORMER.register_module()
class RotatedDeformableDetrTransformer(Transformer):
    """Implements the DeformableDETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 as_two_stage=False,
                 num_feature_levels=5,
                 two_stage_num_proposals=300,
                 **kwargs):
        super(RotatedDeformableDetrTransformer, self).__init__(**kwargs)
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dims = self.encoder.embed_dims
        self.init_layers()


    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans = nn.Linear(self.embed_dims*2,
                                       self.embed_dims*2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims*2)
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dims)
            self.two_stage_wh_embedding = nn.Embedding(1, 2)
            self.two_stage_wh1_embedding = nn.Embedding(1,512)
            self.two_stage_wh2_embedding = nn.Embedding(1,2048)
            self.two_stage_wh3_embedding = nn.Embedding(1,8192)
            self.two_stage_wh4_embedding = nn.Embedding(1,32768)
            self.two_stage_theta1_embedding = nn.Embedding(1,256)
            self.two_stage_theta2_embedding = nn.Embedding(1,1024)
            self.two_stage_theta3_embedding = nn.Embedding(1,4096)
            self.two_stage_theta4_embedding = nn.Embedding(1,16384)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.reference_points = nn.Linear(self.embed_dims, 2)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)
        # nn.init.constant_(self.two_stage_wh_embedding.weight,math.log(0.05 / (1 - 0.05)))
        # nn.init.uniform_(self.two_stage_wh1_embedding.weight, a=math.log(0.05 / (1 - 0.05)),b=math.log(0.08 / (1 - 0.08)))
        # nn.init.uniform_(self.two_stage_wh2_embedding.weight, a=math.log(0.05 / (1 - 0.05)),b=math.log(0.08 / (1 - 0.08)))
        # nn.init.uniform_(self.two_stage_wh3_embedding.weight, a=math.log(0.04 / (1 - 0.04)),b=math.log(0.05 / (1 - 0.05)))
        # nn.init.uniform_(self.two_stage_wh4_embedding.weight, a=math.log(0.04 / (1 - 0.04)),b=math.log(0.05 / (1 - 0.05)))
        nn.init.uniform_(self.two_stage_wh1_embedding.weight, a=0.40,b=0.70)
        nn.init.uniform_(self.two_stage_wh2_embedding.weight, a=0.20,b=0.40)
        nn.init.uniform_(self.two_stage_wh3_embedding.weight, a=0.10,b=0.15)
        nn.init.uniform_(self.two_stage_wh4_embedding.weight, a=0.03,b=0.07)
        nn.init.uniform_(self.two_stage_theta1_embedding.weight, a=0.0,b=(np.pi/2))
        nn.init.uniform_(self.two_stage_theta2_embedding.weight, a=0.0,b=(np.pi/2))
        nn.init.uniform_(self.two_stage_theta3_embedding.weight, a=0.0,b=(np.pi/2))
        nn.init.uniform_(self.two_stage_theta4_embedding.weight, a=0.0,b=(np.pi/2))

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes, learnedwh1=None, learnedwh2=None, learnedwh3=None,
                                     learnedwh4=None, learnedtheta1=None ,learnedtheta2=None, 
                                     learnedtheta3=None, learnedtheta4=None):
        """Generate proposals from encoded memory.

        Args:
            memory (Tensor) : The output of encoder,
                has shape (bs, num_key, embed_dim).  num_key is
                equal the number of points on feature map from
                all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (bs, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).

        Returns:
            tuple: A tuple of feature map and bbox prediction.

                - output_memory (Tensor): The input of decoder,  \
                    has shape (bs, num_key, embed_dim).  num_key is \
                    equal the number of points on feature map from \
                    all levels.
                - output_proposals (Tensor): The normalized proposal \
                    after a inverse sigmoid, has shape \
                    (bs, num_keys, 4).
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0
        # 生成网格一样的proposals
        # spatial_shapes = 16x16, 32x32, 64x64, 128x128
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
                N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            # wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            # grid point 마다 wh를 learn할 수 있도록 생성
            # 각 layer level 별로 embedding을 줘서 학습?
            # wh를 hxx로 줘서 학습가능하게?
            
            if H == 16:
                
                learnedwh1 = torch.reshape(learnedwh1,(16,16,2))
                
                learnedwh1 = learnedwh1.unsqueeze(0)
                learnedwh1 = learnedwh1.repeat(N,1,1,1)
                
                # learnedtheta1 = torch.reshape(learnedtheta1,(16,16,1))
                # learnedtheta1 = learnedtheta1.unsqueeze(0)
                # learnedtheta1 = learnedtheta1.repeat(N,1,1,1)

                # wh  = torch.mul(torch.ones_like(grid),(learnedwh1.sigmoid() * (2.0**lvl)))
                wh  = torch.mul(torch.ones_like(grid),(learnedwh1))
                
                # angle = learnedtheta1
                # print(wh[4][15])
                   
            if H == 32:
                # learnedwh2 = self.two_stage_wh2_embedding.weight[0]
                
                learnedwh2 = torch.reshape(learnedwh2,(32,32,2))
                learnedwh2 = learnedwh2.unsqueeze(0)
                learnedwh2 = learnedwh2.repeat(N,1,1,1)

                # learnedtheta2 = torch.reshape(learnedtheta2,(32,32,1))
                # learnedtheta2 = learnedtheta2.unsqueeze(0)
                # learnedtheta2 = learnedtheta2.repeat(N,1,1,1)
               
                # wh  = torch.mul(torch.ones_like(grid),(learnedwh2.sigmoid() * (2.0**lvl)))
                wh  = torch.mul(torch.ones_like(grid),(learnedwh2))
                # angle = learnedtheta2
                # print('------------------------------')
                # print(wh[4][15])
                # print('------------------------------')
            if H == 64:
                # learnedwh3 = self.two_stage_wh3_embedding.weight[0]
                
                learnedwh3 = torch.reshape(learnedwh3,(64,64,2))
                learnedwh3 = learnedwh3.unsqueeze(0)
                learnedwh3 = learnedwh3.repeat(N,1,1,1)
               
                # learnedtheta3 = torch.reshape(learnedtheta3,(64,64,1))
                # learnedtheta3 = learnedtheta3.unsqueeze(0)
                # learnedtheta3 = learnedtheta3.repeat(N,1,1,1)
               
                # wh  = torch.mul(torch.ones_like(grid),(learnedwh3.sigmoid() * (2.0**lvl)))
                wh  = torch.mul(torch.ones_like(grid),(learnedwh3))
                # angle = learnedtheta3

            if H == 128:
                # learnedwh4 = self.two_stage_wh4_embedding.weight[0]
                
                learnedwh4 = torch.reshape(learnedwh4,(128,128,2))
                
                learnedwh4 = learnedwh4.unsqueeze(0)
                learnedwh4 = learnedwh4.repeat(N,1,1,1)
                
                # learnedtheta4 = torch.reshape(learnedtheta4,(128,128,1))
                # learnedtheta4 = learnedtheta4.unsqueeze(0)
                # learnedtheta4 = learnedtheta4.repeat(N,1,1,1)
                
                # wh  = torch.mul(torch.ones_like(grid),(learnedwh4.sigmoid() * (2.0**lvl)))
                wh  = torch.mul(torch.ones_like(grid),(learnedwh4))
                
                # angle = learnedtheta4
                
            # if H != 16:
            #     wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0**lvl)

            # wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            angle = torch.zeros_like(mask_flatten_)
            proposal = torch.cat((grid, wh, angle), -1).view(N, -1, 5)
            # proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals[..., :4] > 0.01) &
                                  (output_proposals[..., :4] < 0.99)
                                  ).all(
            -1, keepdim=True)
        # output_proposals_valid = ((output_proposals[..., :4] > 0.01) &
        #                           (output_proposals[..., :4] < 0.99)&
        #                           (output_proposals[..., 4] > 0.00)&
        #                           (output_proposals[..., 4] < (np.pi/2))
        #                           ).all(
        #     -1, keepdim=True)
        # output_proposals_valid1 = ((output_proposals[...,5] > 0.00) &
        #                            (output_proposals[...,5] < np.pi/2)).all(
            # -1, keepdim=True)
        # 反sigmoid函数 inversigmoid
        output_proposals[..., :4] = torch.log(output_proposals[..., :4].clone() / (1 - output_proposals[..., :4].clone()))
        # output_proposals = output_proposals.masked_fill(
        #     memory_padding_mask.unsqueeze(-1), float('inf'))
        # output_proposals = output_proposals.masked_fill(
        #     ~output_proposals_valid, float('inf'))
        output_proposals[..., :4] = output_proposals[..., :4].masked_fill(
            memory_padding_mask.unsqueeze(-1), 10000)
        output_proposals[..., :4] = output_proposals[..., :4].masked_fill(
            ~output_proposals_valid, 10000)
        
        
        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    ##### theta 고려해서 position encoding 처리필요
    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals[:,:,:4] = proposals[:,:,:4].sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos


    # query position만 처리 query content는 
    # def get_proposal_pos_embed(self,
    #                            proposals,
    #                            num_pos_feats=64,
    #                            temperature=10000):
    #     """Get the position embedding of proposal."""
    #     scale = 2 * math.pi
    #     dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
    #     dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats)
    #     x_embed = proposals[:, :, 0] * scale
    #     y_embed = proposals[:, :, 1] * scale
    #     pos_x = x_embed[:, :, None] / dim_t
    #     pos_y = y_embed[:, :, None] / dim_t
    #     pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    #     pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    #     if proposals.size(-1) == 2:
    #         pos = torch.cat((pos_y, pos_x), dim=2)
    #     elif proposals.size(-1) == 4:
    #         w_embed = proposals[:, :, 2] * scale
    #         pos_w = w_embed[:, :, None] / dim_t
    #         pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    #         h_embed = proposals[:, :, 3] * scale
    #         pos_h = h_embed[:, :, None] / dim_t
    #         pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

    #         pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    #     else:
    #         raise ValueError("Unknown pos_tensor shape(-1):{}".format(proposals.size(-1)))
    #     return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                bbox_coder=None,
                reg_branches=None,
                cls_branches=None,
                first_stage=False,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.


        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            # pos_embed.shape = [2, 256, 128, 128]
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # [bs, w*h, c]
            feat = feat.flatten(2).transpose(1, 2)
            # [bs, w*h]
            mask = mask.flatten(1)
            # [bs, w*h]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        # multi-scale reference points
        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        # 21760 = 128*128+64*64+32*32+16*16 query的个数
        # memory是编码后的每个query和keys在多层featuremap中对应的特征 一维特征 256
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        
        if self.as_two_stage:
            # input_hw = self.two_stage_wh_embedding.weight[0]
            learnedwh1 = self.two_stage_wh1_embedding.weight[0]
            learnedwh2 = self.two_stage_wh2_embedding.weight[0]
            learnedwh3 = self.two_stage_wh3_embedding.weight[0]
            learnedwh4 = self.two_stage_wh4_embedding.weight[0]
            # learnedtheta1 = self.two_stage_theta1_embedding.weight[0]
            # learnedtheta2 = self.two_stage_theta2_embedding.weight[0]
            # learnedtheta3 = self.two_stage_theta3_embedding.weight[0]
            # learnedtheta4 = self.two_stage_theta4_embedding.weight[0]
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes, learnedwh1, learnedwh2,
                    learnedwh3,learnedwh4)

            # cls score,reg output feature map 별로 짜르기
            enc_outputs_class = cls_branches[self.decoder.num_layers](
                output_memory)
            cls_wieght = cls_branches[self.decoder.num_layers].weight
            enc_outputs_coord_unact_angle= \
                reg_branches[
                    self.decoder.num_layers](output_memory) + output_proposals

            # if first_stage:
            #     return enc_outputs_coord_unact_angle

            topk = self.two_stage_num_proposals
            # topk_proposals = torch.topk(
            #     enc_outputs_class[..., 0], topk, dim=1)[1]

            # enc_outputs_class split, topk_proposals

            
            topk_proposals = torch.topk(
                enc_outputs_class.max(dim=2)[0], topk, dim=1)[1]
            


            # topk_proposals_0 = torch.topk(
            #     enc_outputs_class[..., 0], 25, dim=1)[1]
            # topk_proposals_00 = topk_proposals_0.split(1,dim=0)
            

            # for i in range(bs):
            #     topk_proposals_0_inx = topk_proposals_00[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_0_inx,:] = -99


            # topk_proposals_1 = torch.topk(
            #     enc_outputs_class[..., 1], 6, dim=1)[1]
            # # topk_proposalsss = torch.cat([topk_proposals_0,topk_proposals_1],dim=1)
            # topk_proposals_01 = topk_proposals_1.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_1_inx = topk_proposals_01[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_1_inx,:] = -99

            # topk_proposals_2 = torch.topk(
            #     enc_outputs_class[..., 1], 10, dim=1)[1]
            # topk_proposals_02 = topk_proposals_2.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_2_inx = topk_proposals_02[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_2_inx,:] = -99

            # topk_proposals_3 = torch.topk(
            #     enc_outputs_class[..., 1], 6, dim=1)[1]
            # topk_proposals03 = topk_proposals_3.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals3_inx = topk_proposals03[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals3_inx,:] = -99
            
            # topk_proposals_4 = torch.topk(
            #     enc_outputs_class[..., 1], 28, dim=1)[1]
            # topk_proposals_04 = topk_proposals_4.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_4_inx = topk_proposals_04[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_4_inx,:] = -99

            # topk_proposals_5 = torch.topk(
            #     enc_outputs_class[..., 1], 28, dim=1)[1]
            # topk_proposals_05 = topk_proposals_5.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_5_inx = topk_proposals_05[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_5_inx,:] = -99
            
            # topk_proposals_6 = torch.topk(
            #     enc_outputs_class[..., 1], 50, dim=1)[1]
            # topk_proposals_06 = topk_proposals_6.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_6_inx = topk_proposals_06[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_6_inx,:] = -99
            
            # topk_proposals_7 = torch.topk(
            #     enc_outputs_class[..., 1], 6, dim=1)[1]
            # topk_proposals_07 = topk_proposals_7.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_7_inx = topk_proposals_07[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_7_inx,:] = -99

            # topk_proposals_8 = torch.topk(
            #     enc_outputs_class[..., 1], 6, dim=1)[1]
            # topk_proposals_08 = topk_proposals_8.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_8_inx = topk_proposals_08[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_8_inx,:] = -99

            # topk_proposals_9 = torch.topk(
            #     enc_outputs_class[..., 1], 28, dim=1)[1]
            # topk_proposals_09 = topk_proposals_9.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_9_inx = topk_proposals_09[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_9_inx,:] = -99

            # topk_proposals_10 = torch.topk(
            #     enc_outputs_class[..., 1], 10, dim=1)[1]
            # topk_proposals_010 = topk_proposals_10.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_10_inx = topk_proposals_010[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_10_inx,:] = -99

            # topk_proposals_11 = torch.topk(
            #     enc_outputs_class[..., 1], 6, dim=1)[1]
            # topk_proposals_011 = topk_proposals_11.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_11_inx = topk_proposals_011[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_11_inx,:] = -99

            # topk_proposals_12 = torch.topk(
            #     enc_outputs_class[..., 1], 28, dim=1)[1]
            # topk_proposals_012 = topk_proposals_12.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_12_inx = topk_proposals_012[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_12_inx,:] = -99

            # topk_proposals_13 = torch.topk(
            #     enc_outputs_class[..., 1], 6, dim=1)[1]
            # topk_proposals_013 = topk_proposals_13.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_13_inx = topk_proposals_013[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_13_inx,:] = -99

            # topk_proposals_14 = torch.topk(
            #     enc_outputs_class[..., 1], 7, dim=1)[1]
            # topk_proposals_014 = topk_proposals_14.split(1,dim=0)

            # for i in range(bs):
            #     topk_proposals_14_inx = topk_proposals_014[i].squeeze(0)
            #     enc_outputs_class[i,topk_proposals_14_inx,:] = -99

            # topk_proposals = torch.cat([topk_proposals_0,topk_proposals_1,topk_proposals_2, topk_proposals_3,topk_proposals_4,topk_proposals_5,topk_proposals_6,topk_proposals_7,topk_proposals_8,
            #                             topk_proposals_9,topk_proposals_10,topk_proposals_11,topk_proposals_12,topk_proposals_13,topk_proposals_14], dim=1)


            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact_angle, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 5))
            topk_coords_unact = topk_coords_unact.detach()
           
            # topk_memory

            # topk_memory = torch.gather(
            #     output_memory, 1,
            #     topk_proposals.unsqueeze(-1).repeat(1, 1, 256))
            # topk_memory = topk_memory.detach()
            # quantized, indices, commit_loss = vq(topk_memory)

            # topk_class_argmax = enc_outputs_class.argmax(2)
            # # topk_class_argmax = torch.as_tensor(
            # # topk_class_argmax, dtype=torch.long, device=enc_outputs_class.device)
            # topk_class_onehot = torch.zeros(enc_outputs_class.shape, device = enc_outputs_class.device)
            # topk_class_onehot = topk_class_onehot.scatter(2,topk_class_argmax.unsqueeze(2), 1.0)
            
            # topk_class_onehot = torch.gather(
            #     topk_class_onehot, 1,
            #     topk_proposals.unsqueeze(-1).repeat(1, 1, 15))
            # topk_class_onehot = topk_class_onehot.detach()

            # topk_class = torch.gather(
            #     output_memory, 1,
            #     topk_proposals.unsqueeze(-1).repeat(1, 1, 256))
            # topk_class = topk_class.detach()

            # quantized, indices, commit_loss = vq(topk_class, topk_class_onehot)
            # quantized = quantized.detach()
            # obb2xyxy
            # reference_points = obb2poly_tr(topk_coords_unact).sigmoid()
            
            
            reference_points = topk_coords_unact[..., :4].sigmoid()

            ##### theta 고려해서 reference points를 5개 줘야함
            # reference_points = torch.cat(topk_coords_unact[...,:4].sigmoid(),topk_coords_unact[4])

            init_reference_out = reference_points
            # obb2xywh

            ##### theta 고려해서 norm 필요
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact[..., :4])))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
            
            # topk_class = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1)
            # query = topk_class
            # query_pos = self.pos_trans_norm(
            #     self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact[..., :4])))
             
            # query = query+quantized
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        # decoder
        # if first_stage:
        # vq = VectorQuantize(
        #     dim=256,
        #     codebook_size = 15,
        #     codebook_dim = 256,
        #     # orthogonal_reg_weight = 10,
        #     # kmeans_init= True,
        #     sync_codebook = False,  
        # )
        
        query = query.permute(1, 0, 2)
        # if not first_stage:
        # quantized, indices, commit_loss = vq(topk_class, topk_class_onehot)
        # print(commit_loss)
        # quantized = quantized.permute(1, 0, 2)
        # query = torch.add(query,quantized)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            bbox_coder=bbox_coder,
            **kwargs)

        inter_references_out = inter_references
        if self.as_two_stage:
            return inter_states, init_reference_out, \
                   inter_references_out, enc_outputs_class, \
                   enc_outputs_coord_unact_angle
        return inter_states, init_reference_out, \
               inter_references_out, None, None

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RotatedDeformableDetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):

        super(RotatedDeformableDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                bbox_coder=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        ##### theta 고려해서 reference points 5개로 각도는 valid ratios 고려 x
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                                         torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                                         valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                # tmp = obb2xyxy(reg_branches[lid](output), version='le90')
                tmp = reg_branches[lid](output)
                ###### reference points 5개로 theta고려 필요
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp[..., :4] + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                                                    ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points
