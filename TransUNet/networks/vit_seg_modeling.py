# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


# convert HWIO to OIHW
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


# 一种新型的激活函数 g(x) = x * softmax(x)
def swish(x):
    return x * torch.sigmoid(x)


# 三种激活函数 gelu、relu、switch

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# 注意力机制
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]  # 注意力的头的数量:12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768
        # q,k,v  Linear(in_features:隐藏的大小，out_features:所有头的大小=头的数量*头的大小)
        self.query = Linear(config.hidden_size, self.all_head_size)  # hidden_size=768，all_head_size=768
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        # 输出： 隐藏层大小
        self.out = Linear(config.hidden_size, config.hidden_size)
        # 丢弃率
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        # dim=-1 是对最后一个维度进行softmax
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # x.size()[:-1] + (12,64) -> (2,325,12,64)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # * 可能是解包操作去掉tuple属性， new_x_shape是一个torch.Size()，x.view()之后就把他变成了一个tensor
        return x.permute(0, 2, 1, 3)  # 改变一下维度  (2,12,325,64)

    def forward(self, hidden_states):
        # hidden_states -> [batch_size, num_patches, total_dim] -> (2, 325, 768),325=324+1，是嵌入向量。
        # self.query\key\value是各自的映射，输出为(2,325,768)。此处的768=12*64是12个注意力抽头，每个是64维向量
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # （2，12，325，64）12表示12个注意力，325是嵌入向量的个数。计算时，计算每个注意力抽头中的不同嵌入向量间的关系，
        # 最后将12个注意力的输出结果进行合并，因此此处通过transpose_for_scores函数进行reshape
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # 计算相似度  Q * (K^T)-转置     输出后两维度的每一行表示所有嵌入向量与当前向量的相似度
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (2,12,325,325)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # / sqrt(d_k)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None  # (2,12,325,325)
        attention_probs = self.attn_dropout(attention_probs)
        # 给各个向量分配权重，q*kT*v
        context_layer = torch.matmul(attention_probs, value_layer)  # (2,12,325,64)
        # 合并所有的注意力头
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (2,325,12,64)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # (2,325,768)
        attention_output = self.out(context_layer)  # (2,325,768)
        attention_output = self.proj_dropout(attention_output)
        # 输出的是各个向量计算多个注意力抽头，分配权重，合并抽头，并映射为嵌入维度的向量、12个heads对应的vectors间的相似度关系
        return attention_output, weights


# 多层感知机  实现的是注意力输出结果，进行映射
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    # 初始化权重
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)  # 均匀分布
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)  # 正态分布
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)  # 激活函数 gelu
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# 将图片embedding,并加上对应的位置信息
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None  # 混合模型
        self.config = config
        img_size = _pair(img_size)  # _pair 返回(img_size,img_size)

        if config.patches.get("grid") is not None:  # ResNet50
            grid_size = config.patches["grid"]  # (16,16)
            # patch_size = (512 / 16 / 16 = 2, 2)  不重叠的获取，此时分割线处的内容信息就不能完整获得
            # img_size[0] // 16 这里的16是指图片经过resnet50之后H变为原来的1/16.   再除以网格大小就得到了patch_size
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            # patch_size_real = (32,32)
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16) # （32,32）
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  # n_patches = 16 * 16
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])  # (16,16) or (32,32)
            # N(图片批次个数) = (H * W) / P^2
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 32 * 32
            self.hybrid = False
        # hybrid混合
        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16  # 64 * 16 = 1024 也就是最后一层resnet的输出通道数
        # kernel和stride都是patch_size大小: 表示不重叠的获取，也就是把图片分成一个一个的网格
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # 位置信息
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))  # [[ [hidden_size个数], ]]

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)  # ResNetV2(x)  [B,C,H,W]=[B,1024,512/16=32]
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)  # flatten降维 [B,C,H,W] => [B,C, HW]
        x = x.transpose(-1, -2)  # => [B, HW, C] / (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


# 一个Transformer Layer块  架构图最左边的那个图
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)  # LayerNorm
        x, weights = self.attn(x)  # 架构中的MSA
        x = x + h  # 残差连接

        h = x
        x = self.ffn_norm(x)  # LayerNorm
        x = self.ffn(x)  # MLP
        x = x + h
        # weights是每个块内注意力的参数，也就是12个注意力抽头的嵌入向量间的余弦相似度矩阵
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # np2th : convert HWIO to OIHW
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)  # 相当于直接拉长变为一维tensor
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


# 编码器
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):  # 12个Transformer Layer
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


# 图中将hidden feature [D, H/16, W/16] -> [512,H/16,W/16]那一步
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


# 单个解码块-对应图像连接cnn部分+卷积部分+上采样部分
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,  # 加上前面CNN连接过来的
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # 上采样---相当于图像插值 scale_factor=2，变为原来的2倍

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# 最后一步，还原成图像那个地方
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(  # 对由多个输入通道组成的输入信号进行二维双线上采样。
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()  # nn.Identity()一个占位符
        super().__init__(conv2d, upsampling)


# 整个解码部分
class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])   # [512, 256, 128, 64]
        out_channels = decoder_channels  # (256, 128, 64, 16)

        if self.config.n_skip != 0:  # n_skip = 3
            skip_channels = self.config.skip_channels  # [512, 256, 64, 16]
            for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0  # 不就是吧skip_channels变为 [512, 256, 64, 0]

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        # in:[512, 256, 128, 64]  out:[256, 128, 64, 16]    skip:[512, 256, 64, 0]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)  # reshape: (n_patch,D) -> (D,H/16,W/16)
        x = x.contiguous().view(B, hidden, h, w)  # contiguous()使之连续方便使用view()
        x = self.conv_more(x)  # Conv2dReLU
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    # num_classes=21843 ？这个数字好像..不知道哪里来的，而且好像并没有用上..
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()  # 初始化继承的属性
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier  # 'seg'(该模式好像一般用于预训练) or 'token'
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],   # c_classes = 2输出通道
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # 中间那个维度重复三次  即有原来的 [B,C,H,W] -> [B,3*C,H,W]
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)  # DecoderCup
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:  # pos 比 new_pos多一个维度
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)  # size(1) 获取size的第二个维度，如[1,256,512], 则size(1) = 256
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]   # [0,1:]表示如果原来的维度是【1,256,512】  -> [255,512]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np   order=1:双线性插值 0：最近邻插值
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}
