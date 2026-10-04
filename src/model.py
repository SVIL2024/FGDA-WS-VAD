from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        """
        对视频进行编码处理，将图像序列转换为特征表示
        
        参数:
            images: 输入的图像张量，形状为(batch_size, sequence_length, feature_dim)
            padding_mask: 填充掩码，用于标识有效数据位置
            lengths: 每个序列的实际长度
            
        返回:
            x: 编码后的特征张量，经过图卷积和线性变换处理
        """
        images = images.to(torch.float)
        # 计算位置嵌入并添加到图像特征中
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        # 使用位置嵌入层将离散的位置ID转换为连续的向量表示
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        # 调整张量维度顺序，将序列长度维度移到最前面
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        # 将位置嵌入添加到图像特征上，为每个时间步的图像特征注入位置信息
        images = images.permute(1, 0, 2) + frame_position_embeddings
        # 通过时间维度处理获取时序特征
        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        # 构建邻接矩阵并进行图卷积操作
        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        # 进行第二层图卷积处理
        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        # 拼接两种图卷积结果并通过线性层输出
        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def encode_textprompt(self, text):
        """
        对文本提示进行编码，将文本转换为CLIP模型可处理的嵌入表示
        
        Args:
            text: 输入的文本数据，用于生成文本嵌入
            
        Returns:
            text_features: 经过CLIP模型编码后的文本特征向量
        """
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        # 构建文本嵌入：将单词嵌入按照特定位置规则插入到预定义的嵌入模板中
        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        # 使用CLIP模型对构建好的文本嵌入进行最终编码
        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def forward(self, visual, padding_mask, text, lengths):
        """
        前向传播函数，执行视频-文本匹配任务
        
        Args:
            visual: 视频输入数据
            padding_mask: 视频填充掩码
            text: 文本输入数据
            lengths: 序列长度信息
            
        Returns:
            tuple: 包含以下元素
                - text_features_ori: 原始文本特征
                - logits1: 第一个分类器输出的logits
                - logits2: 最终的相似度分数矩阵
        """
        # 编码视频特征并生成初步分类结果
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        # 编码原始文本特征
        text_features_ori = self.encode_textprompt(text)

        text_features = text_features_ori
        # 计算注意力权重并进行特征融合，
        # 将 [batch_size, sequence_length, 1] 的logits重新排列为 [batch_size, 1, sequence_length]
        logits_attn = logits1.permute(0, 2, 1) 
        # 使用矩阵乘法将注意力权重应用于视觉特征
        visual_attn = logits_attn @ visual_features
        # 对注意力后的视觉特征进行L2归一化，使特征向量长度为1
        # 这有助于稳定训练过程和提高模型性能
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        # 将视觉注意力特征扩展，使其第一个维度为batch_size，第二个维度为文本特征数量，第三个维度保持不变
        # 这样做是为了让每个视频能与多个文本进行比较

        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        
        # 自适应加权: 根据视频内容的重要程度调整文本表示
        # 双向增强: 视觉信息增强文本表示，同时保持文本语义完整性
        
        # 首先增加一个新的维度（在第0维）
        # 然后扩展维度以匹配视觉注意力特征的形状
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        # 将原始文本特征与注意力增强的视觉特征相加
        text_features = text_features + visual_attn
        # 再加上MLP变换后的特征，实现非线性特征变换
        # 多层变换: 通过MLP进一步提取融合特征的高级模式
        text_features = text_features + self.mlp1(text_features)

        # 计算归一化后的视觉和文本特征相似度，对视觉特征进行L2归一化，使每个时间步的特征向量长度为1
        # dim=-1表示沿最后一个维度（特征维度）计算范数
        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        # 对经过注意力增强的文本特征进行L2归一化，视频每个时间步特征与所有文本特征的余弦相似度
        # visual_features_norm: [batch_size, seq_len, features]
        # text_features_norm:   [batch_size, features, num_texts]
        # logits2:              [batch_size, seq_len, num_texts]
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        # 将文本特征从[batch_size, num_texts, features]转换为[batch_size, features, num_texts]
        text_features_norm = text_features_norm.permute(0, 2, 1)
        # 通过矩阵乘法计算视觉-文本特征相似度，CLIP系列模型: 通常在 0.01 到 0.5 之间，原始论文中使用 0.07
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        return text_features_ori, logits1, logits2
    