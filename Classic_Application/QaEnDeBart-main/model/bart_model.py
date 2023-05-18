from typing import Dict, List, Optional, Tuple
import math
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_bart import BartConfig   # 后续可能会进行改变
from transformers.activations import ACT2FN
from model.loss import LabelSmoothedCrossEntropyCriterion

def invert_mask(attention_mask):
    assert attention_mask.dim() == 2  # padding mask (bsz, seq_length)
    return attention_mask.eq(0)   # 进行mask_select的操作


def _prepare_bart_decoder_inputs(
        config,
        input_ids,
        decoder_input_ids=None,
        decoder_padding_mask=None,
        causal_mask_dtype=torch.float32
):
    r"""
    Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    :param config:
    :param input_ids: 实际上就是target(summary)
    :param decoder_input_ids:
    :param decoder_padding_mask:
    :param causal_mask_type:
    :return:对decoder的输入进行处理。target padding操作，右移一位，下三角的attention_mask
            decoder_input_ids:经过了右移操作之后的target,用于teacher forcing的训练
            decoder_padding_mask:处理paddin token，得到整个输入序列的bool,padding的位置为True
            causal mask:上三角的mask矩阵，下三角全部为0，上三角全部为-inf,用于生成任务
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)

    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)  # 得到bool类型的mask矩阵
    # 得到上三角全部是-inf的矩阵，-inf通过softmax变成0
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask


class PretrainedBartModel(PreTrainedModel):
    config_calss = BartConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    @property
    def dummy_inputs(self):
        """如果数据缺失的话使用这个代替"""
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }

        return dummy_inputs


def _make_linear_from_emb(embed):
    """将线性层权重设置为embeds的权重， 得到了一个 embedding的线性层函数"""
    vocab_size, emb_size = embed.weight.shape
    linear_layer = nn.Linear(vocab_size, emb_size, bias=False)
    linear_layer.weight.data = embed.weight.data
    return linear_layer


def _check_shapes(shape1, shape2):
    if shape1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape1, shape2 ))


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
    将输入的token进行padding操作，并且使用eos进行结尾，将token向右移动一个token
    input_ids: tensor([[ 1,  2,  3,  4,  5,  6,  0,  0],
        [10, 11, 12, 13, 14, 15, 16,  0]])
    pre_output_tokens；
    [[ 6,  1,  2,  3,  4,  5,  6,  0],
        [16, 10, 11, 12, 13, 14, 15, 16]])
        实际上就是用于teacher forcing训练
    """
    pre_output_tokens = input_ids.clone()
    # 获取batch中每一个句子非pad的个数-1，即最后一个非pad token的索引
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    # ganther操作沿着指定的维度获取Tensor,dim=1表示行固定
    pre_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    pre_output_tokens[:, 1:] = input_ids[:, :-1]
    return pre_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask   # True部分为padding token的部分


def fill_with_neg_inf(tensor):
    """FP16-compatible function that fills a input_ids with -inf """
    return tensor.float().fill_(float("-inf")).type_as(tensor)


# 一层transformer的encoder
class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super(EncoderLayer, self).__init__()
        self.embed_dim = config.d_model   # hidden_state的维度
        self.output_attentions = config.output_attentions   # 表示是否输出attention的信息
        self.self_attn = SelfAttention(
            self.embed_dim,
            config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.normalize_before = config.normalize_before   # 表示是否再输入每一层transformer之前进行norm
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)   # 输出之后的layer norm

    def forward(self, x, encoder_padding_mask):
        """
                Args:
                    x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
                    encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                        `(batch, src_len)` where padding elements are indicated by ``1``.
                    for t_tgt, t_src is excluded (or masked out), =0 means it is
                    included in attention  padding的位置为true  不是padding的位置为fasle

                Returns:
                    encoded output of shape `(seq_len, batch, embed_dim)`
                """
        residual = x
        if self.normalize_before:  # before表示的是在self-attention之前进行layerNorm操作
            x = self.normalize_before(x)
        x, attn_weights = self.self_attn(
            query=x,
            key=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=self.output_attentions,

        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)   # 在self-attenion之后做layer-norm操作

        # FFN
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn_weights


class BartEncoder(nn.Module):
    """
        Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
        is a :class:`EncoderLayer`.    embed_tokens表示的是初始化的embedding函数 实际上就是 nn.embedding(vocab_size, embedding_dim)

        Args:
            config: BartConfig
        """
    def __init__(self, config, embed_tokens):
        super(BartEncoder, self).__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop     # layerdrop表示将transformer层进行skip的阈值
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings
        self.output_attentions = config.output_attentions
        self.embed_tokens = embed_tokens   # input_ids embedding的映射函数
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                embed_dim,
                self.padding_idx,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        # 是对embedding进行layer norm
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None
        self.segment_embeddings = nn.Embedding(2, config.d_model)    # 加入了segment部分

    def forward(
            self,
            input_ids,
            attention_mask,
            segment_ids=None,
    ):

        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *self.output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embeds_pos = self.embed_positions(input_ids)
        if segment_ids is not None:
            segment_embeds = self.segment_embeddings(segment_ids)
            x = inputs_embeds + embeds_pos + segment_embeds  # 注意因为这里加入了问答的部分 因此将query和document进行区别 加入了segment embedding的部分
        else:
            x = inputs_embeds + embeds_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B * T * C -->T * B * C    ---> seq_len, batch_size, embed_dim
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []  # 所有层的encoder states和attention 权重
        for encoder_layer in self.layers:
            if self.output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop
            dropout_probabilty = random.uniform(0, 1)
            if self.training and (dropout_probabilty < self.layerdrop):  # skip the layer   当layerdrop设置为0的时候不会去掉任何一层的
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask)

            if self.output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:  # 表示最后一层是否进行layer-norm的操作
            x = self.layer_norm(x)
        if self.output_hidden_states:
            encoder_states.append(x)   # 将最后一层的隐状态输出

        # T * B * c -> B * T * C 再变回到batch_size, seq_len, embedding_size
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)   # 最后输出的隐状态

        return x, encoder_states, all_attentions   # 最后一层的隐状态， 每一层的隐状态，每一层的多头的attention


class DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super(DecoderLayer, self).__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
            self,
            x,
            encoder_hidden_states,
            encoder_attn_mask=None,
            layer_state=None,
            causal_mask=None,
            decoder_padding_mask=None,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Self Attention
        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            need_weights=self.output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross Attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # FFN
        residual = x
        if self.normalize_before:
            x = self.fianl_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.fianl_layer_norm(x)

        # # just self_attn weights for now, following t5, layer_state = cache for decoding
        return (
            x,
            self_attn_weights,  # 是decoder端的attention的权重
            layer_state,
        )


class BartDecoder(nn.Module):
    """
        Transformer decoder consisting of *config.decoder_layers* layers. Each layer
        is a :class:`DecoderLayer`.
        Args:
            config: BartConfig
            embed_tokens (torch.nn.Embedding): output embedding
        """
    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super(BartDecoder, self).__init__()
        self.output_attention = config.output_attentions,
        self.output_hidden_states = config.output_hidden_states
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                config.pad_token_id,
            )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )

        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
            self,
            input_ids,
            encoder_hidden_states,
            encoder_padding_mask,
            decoder_padding_mask,
            decoder_causal_mask,
            decoder_cached_states=None,
            use_cache=False,
            **unused
    ):
        """
       Includes several features from "Jointly Learning to Align and
       Translate with Transformer Models" (Garg et al., EMNLP 2019).

       Args:
           input_ids (LongTensor): previous decoder outputs of shape
               `(batch, tgt_len)`, for teacher forcing
           encoder_hidden_states: output from the encoder, used for
               encoder-side attention
           encoder_padding_mask: for ignoring pad tokens
           decoder_cached_states (dict or None): dictionary used for storing state during generation

       Returns:
           tuple:
               - the decoder's features of shape `(batch, tgt_len, embed_dim)`
               - hidden states
               - attentions
        """
        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)    # 将padding mask矩阵转换为bool mask矩阵。padding的位置为True

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Barot output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # deocder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []

        for idx, decoder_layer in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states += (x, )   # 第一层，加入的是embedding的表示
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers - 1)):   # mbart最后一层
                x = self.layer_norm(x)
            if self.output_attention:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        return x, next_cache, all_hidden_states, list(all_self_attns)


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions, embedding_dim, padding_index=None):
        super(SinusoidalPositionalEmbedding, self).__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding dim {embedding_dim} not supported")
        self.weight = self._init_weight(self.weight)   # self.weight是nn.embedding中的权重      源码中self.weighting = Parameter(torch.tensor(num_embeddings, embedding_dim))

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
                    The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array([pos / pow(10000, 2*(j//2) / dim)for j in range(dim)]for pos in range(n_pos))
        out[:, 0:dim//2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))    # 奇数位置的话使用sin
        out[:, dim//2:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))     # 偶数位置使用cos进行编码
        out.detach_()
        out.requires_grad = False
        return out
    
    @torch.no_grad()       # 修饰符带的那个函数torch.no_grad()的入口参数就是下面的整个函数forward
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len-1)    # new函数构建一个具有相同类型的tensor  自定义指定维度  positions表示最大的位置索引index
        else:
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        
        return super(SinusoidalPositionalEmbedding, self).forward(positions)


class LearnedPositionalEmbedding(nn.Embedding):
    """
        This module learns positional embeddings up to a fixed maximum size.
        Padding ids are ignored by either offsetting based on padding_idx
        or by setting padding_idx to None and ensuring that the appropriate
        position ids are passed to the forward function.
    """
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        assert padding_idx is not None
        num_embeddings += padding_idx + 1   # 指的是每一个pos的位置全部加上padding_idx+1， 因此最长的position的id就变成了num_embeddings+padding_idx+1
        super(LearnedPositionalEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx)
        
    def forward(self, input, use_cache=False):
        if use_cache:
            pos = int(self.padding_idx + input.size(1))
            positions = input.data.new(1, 1).fill_(pos)
        else:
            positions = create_position_from_input_ids(input, self.padding_idx)
        return super(LearnedPositionalEmbedding, self).forward(positions)   # positions表示的是每一个token位置的id


def create_position_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
        padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
        `utils.make_positions`.

        :param torch.Tensor x:
        :return torch.Tensor:
        """
    mask = input_ids.ne(padding_idx).int()
    # torch.cumsum返回沿着指定维度的累积和   如输入的是一个N元的向量，结果也是一个N元向量，第i个输出的元素值为 x1+x2+x3+...xi
    # 对于二维张量来说dim=1表示的是在行的维度上进行累加
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask   # incremental表示原始的索引逐渐增加的结果   *mask之后相当于原始位置为padding 的token全部变成0
    return incremental_indices.long() + padding_idx     # 为什么要加一个padding呢？Position numbers begin at padding_idx+1.


# 需要再仔细看一下
class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            encoder_decoder_attention=False,  # 除了这个就是self-attention
    ):

        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"   # 表示self attention的类型

    def _shape(self, tensor, dim_0, bsz):   # dim_0表示的是seq_len

        return tensor.contiguous().view(dim_0, bsz*self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
           self,
           query,
           key: Optional[Tensor],
           key_padding_mask: Optional[Tensor] = None,
           layer_state: Optional[Dict[str, Optional[Tensor]]] = None, 
           attn_mask: Optional[Tensor] = None,
           need_weights=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention   # 表示encoder_decoder attention, 其中的k v全部来自encoder 隐状态
        tgt_len, bsz, embed_dim = query.size()
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:
            save_state = layer_state.get(self.cache_key, {})  # dict get函数查找字典中对应键值的value,没有找到返回空字典   得到的是关于整个attention的信息
            if "prev_key" in save_state:
               # previous time steps are cached - no need to recompute key and value if they are static
               # 之前步已经保存了key、value并且是静态的，那么不需要重新计算
                if static_kv:  # 如果是encoder-decoder attention,key value的值全部为None 使用static_kv代替
                    key = None
        else:  # 相当于第一层的attention
            save_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if static_kv:    # static_kv表示的是decoder端的encoder-decoder attention,kv都是encoder最后一层的表征
            if key is None:    # encoder_decoder attention
                k = v = None
            else:
                k = self.k_proj(query)  # decoder self-attention
                v = self.v_proj(query)
        else:
            k = self.k_proj(query)  # encoder self-attention
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)   # tgt_len, bach_size * num_head, head_dim
        if k is not None:   # 如果k.v是None 说明当前的attention表示的是encoder-decoder attention
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if save_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, save_state, key_padding_mask, static_kv, bsz)   # 对于第一层transformer 的encoder-decoder来说， k,v是之前缓存的encoder的hidden states

        # Update cache   cache_key有两种情况 1.encoder-decoder 2 self
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part ofsz a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:   # 维度为0表示单个tensor值
            key_padding_mask = None
            assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

        # don't attend to padding symbols
        if key_padding_mask is not None:

            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)   # key_padding_mask原本是2维的 现在增加两维
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        # attention scores
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(
           self,
           k,
           v,
           saved_state,
           key_padding_mask,
           static_kv,
           bsz,
    ):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        # 保存state
        if "pre_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)   # dim=1表示横着拼接 即增加列数
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
            assert k is not None and v is not None
            prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
            key_padding_mask = self._cat_prev_key_padding_mask(
                key_padding_mask, prev_key_padding_mask, bsz, k.size(1), static_kv
            )

            return k, v, key_padding_mask

    @ staticmethod
    def _cat_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)

        elif key_padding_mask is not None:
            filler = torch.zeros(
                batch_size,
                src_len - key_padding_mask.size(1),
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(filler, key_padding_mask, dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask

        return new_key_padding_mask


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def _filter_out_falsey_values(tup) -> Tuple:
    """Remove entries that are None or [] from an iterable."""
    return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)

def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_callable(*docstr):
    def docstring_decorator(fn):
        class_name = ":class:`~transformers.{}`".format(fn.__qualname__.split(".")[0])
        intro = "   The {} forward method, overrides the :func:`__call__` special method.".format(class_name)
        note = r"""

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        pre and post processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
               Indices of input sequence tokens in the vocabulary. Use BartTokenizer.encode to produce them.
            Padding will be ignored by default should you provide it.
            Indices can be obtained using :class:`transformers.BartTokenizer.encode(text)`.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices in input_ids.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
            Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
            `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention of the decoder.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            Provide for translation and summarization training. By default, the model will create this tensor by shifting the input_ids right, following the paper.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
            Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
            If you want to change padding behavior, you should read :func:`~transformers.modeling_bart._prepare_decoder_inputs` and modify.
            See diagram 1 in the paper for more info on the default strategy
"""

BART_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matters related to general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.

"""
BART_GENERATION_EXAMPLE = r"""
    Examples::

        from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
        # see ``examples/summarization/bart/evaluate_cnn.py`` for a longer example
        model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
        ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        inputs = tokenizer.batch_encode_plus([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
        # Generate Summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

"""


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.", BART_START_DOCSTRING,
)
class BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super(BartModel, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.padding_idx, self.vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(self.vocab_size, config.d_model, self.padding_idx)   # 相当于定义初始化的token embedding

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids=None,
            decoder_input_ids=None,
            encoder_outputs: Optional[Tuple] = None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            use_cache=False,
            only_encoder=False,
    ):
        if not use_cache and not only_encoder:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask = None
            causal_mask = None

        if not only_encoder:
            assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, segment_ids=token_type_ids,
                                           attention_mask=attention_mask)
        assert isinstance(encoder_outputs, tuple)

        if only_encoder:
            return encoder_outputs  # 如果只是使用encoder的话那么直接返回

        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )

        # attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs: Tuple = _filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0], torch.Tensor)
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
        return {"encoder_outputs": encoder_outputs, "decoder_outputs": decoder_outputs}   # 返回值和transformers略有不同
        # return decoder_outputs + encoder_outputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.",
    BART_START_DOCSTRING + BART_GENERATION_EXAMPLE,
)
class BartQuestionAnsweringHead(nn.Module):
    def __init__(self, inner_dim):
        super(BartQuestionAnsweringHead, self).__init__()
        # 去掉了线性层之前的 dropout  并且分别使用两个线性层继续预测start end的位置
        self.start_dense = nn.Linear(inner_dim, 1)
        self.end_dense = nn.Linear(inner_dim, 1)
        self.init_linear_weights(self.start_dense)
        self.init_linear_weights(self.end_dense)

    def forward(self, x):
        start_logits = self.start_dense(x)
        end_logits = self.end_dense(x)
        return start_logits, end_logits

    def init_linear_weights(self, m):
        assert isinstance(m, nn.Linear), "don't init linear weights for not Linear Module"
        nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2.0))   # 使用高斯分布进行初始化
        nn.init.constant_(m.bias, 0)


class BartEncoderForQuestionAnswering(PretrainedBartModel):
    def __init__(self, config, args):
        super(BartEncoderForQuestionAnswering, self).__init__(config, args)
        self.num_labels = config.num_labels
        self.model = BartModel.from_pretrained(
            args.model_name_or_path,
            config=config,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            cache_dir=args.cache_dir if args.cache_dir is not None else None,
        )

        # 冻结bart model部分的decoder的参数
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        self.qa_head = BartQuestionAnsweringHead(inner_dim=config.d_model)   # 不需要进行初始化了

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                start_positions=None,
                end_positions=None,
                only_encoder=True):

        for param in self.model.decoder.parameters():
            param.requires_grad = False    # 将decoder端的所有的参数全部冻结   只训练encoder端的部分

        outputs = self.model(
            input_ids,
            attention_mask,
            token_type_ids,
            only_encoder=only_encoder,
        )
        hiddens = outputs[0]
        start_logits, end_logits = self.qa_head(hiddens)
        start_logits = start_logits.squeeze(-1)    # bsz*seq_len
        end_logits = end_logits.squeeze(-1)        # bsz * seq_len
        outputs = (start_logits, end_logits) + outputs[2:]

        ## 计算损失
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.unsqueeze(-1)    # shape:(batch_size, )
            if len(end_positions.size()) > 1:
                end_positions = end_positions.unsqueeze(-1)

            ignored_index = start_logits.size(1)        # 索引越界了
            start_positions.clamp_(0, ignored_index)      # clamp_函数positions_labels的长度控制在序列的长度范围上  超过的位置全部使用seq_len代替
            end_positions.clamp_(0, ignored_index)

            loss_fun = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fun(start_logits, start_positions)
            end_loss = loss_fun(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2.0
            outputs = (total_loss,) + outputs
        return outputs


# 需要考虑到验证集的部分
# class BartForSeq2SeqGeneration(PretrainedBartModel):
#     def __init__(self, config: BartConfig):
#         super(BartForSeq2SeqGeneration, self).__init__(config)
#         base_model = BartModel(config=config)
#         self.model = base_model
#         self.qa_head = BartQuestionAnsweringHead(
#             config.d_model, config.d_model, num_classes=2, pooler_dropout=0.0
#         )
#         self.regitser_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))   # 这个线性层是和embedding层共享参数的，同时也是mask LM 的预测层
#         self.model._init_weights(self.qa_head.dense)
#         self.model._init_weights(self.qa_head.out_proj)
#
#     def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> torch.nn.Embedding:
#         old_num_tokens = self.model.shared.num_embeddings
#         new_embeddings = super(BartForSeq2SeqGeneration, self).resize_token_embeddings(new_num_tokens)
#         self.model.share = new_embeddings
#         self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
#
#         return new_embeddings
#
#     def _resize_final_logits_bias(self, new_num_tokens: int , old_num_tokens: int) -> None:
#         if new_num_tokens <= old_num_tokens:
#             new_bias = self.final_logits_bias[:, new_num_tokens]
#         else:
#             extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
#             new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
#         self.register_buffer("final_logits_bias", new_bias)
#
#     @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
#     def forward(
#             self,
#             input_ids,
#             attention_mask,
#             decoder_input_ids,
#             segment_ids=None,
#             target_labels=None,
#             encoder_output=None,
#             decoder_attention_mask=None,
#             decoder_cached_states=None,
#             start_positions=None,
#             end_positions=None,
#             candidate_start_positions=None,
#             candidate_end_positions=None,
#             masked_labels=None,
#             use_cache=False,
#             **unused,
#     ):
#         r"""
#                 labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
#                     Labels for computing the masked language modeling loss.
#                     Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
#                     Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
#                     with labels
#                     in ``[0, ..., config.vocab_size]``.
#
#             Returns:
#                 :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
#                 masked_lm_loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
#                     Masked language modeling loss.
#                 prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
#                     Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
#                 hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#                     Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the  output of each layer)
#                     of shape :obj:`(batch_size, sequence_length, hidden_size)`.
#
#                     Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#                 attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#                     Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#                     :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
#
#                     Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#                     heads.
#
#             Examples::
#
#                     # Mask filling only works for bart-large
#                     from transformers import BartTokenizer, BartForConditionalGeneration
#                     tokenizer = BartTokenizer.from_pretrained('bart-large')
#                     TXT = "My friends are <mask> but they eat too many carbs."
#                     model = BartForConditionalGeneration.from_pretrained('bart-large')
#                     input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
#                     logits = model(input_ids)[0]
#                     masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
#                     probs = logits[0, masked_index].softmax(dim=0)
#                     values, predictions = probs.topk(5)
#                     tokenizer.decode(predictions).split()
#                     # ['good', 'great', 'all', 'really', 'very']
#                 """
#         if "lm_labels" in unused:
#             warnings.warn(
#                 "The 'lm_labels' argument is deprecate and will be removed in a feature version, use 'labels' instead.",
#                 DeprecationWarning,
#             )
#             target_labels = unused.pop("lm_labels")
#
#         outputs = self.model(
#             input_ids,
#             segment_ids=segment_ids,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             encoder_output=encoder_output,
#             decoder_attention_mask=decoder_attention_mask,
#             decoder_cached_states=decoder_cached_states,
#             use_cache=use_cache,
#         )  # 注意当前的输出得到的是字典的形式   output={"encoder_outputs": encoder_outputs, "decoder_outputs": decoder_outputs}
#         encoder_outputs = outputs["encoder_outputs"]
#         decoder_outputs = outputs["decoder_outputs"]
#         loss_function = nn.CrossEntropyLoss(ignore_index=-100)
#
#         # 计算摘要的损失
#         target_logits = F.linear(decoder_outputs[0], self.model.shared.weight, bias=self.final_logtis_bias)
#         summary_loss = loss_function(target_logits.view(-1, self.config.vocab_size), target_labels.view(-1))
#
#         # 计算问答的损失
#         answer_logits = self.qa_head(encoder_outputs[0])
#         answer_start_loss = loss_function(answer_logits.view(-1, self.config.vocab_size), start_positions.view(-1))
#         answer_end_loss = loss_function(answer_logits.view(-1, self.config.vocab_size), end_positions.view(-1))
#         answer_loss = answer_start_loss + answer_end_loss
#
#         # 计算masked lm的损失
#         # encoder端的masked 预测的线性层与decoder端的预测的线性层共享的，得到每一个masked token的分布
#         masked_logits = F.linear(encoder_outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
#         masked_loss = loss_function(masked_logits.view(-1, self.config.vocab_size), target_labels.view(-1))
#
#         loss = summary_loss + answer_loss + masked_loss
#
#
#         # 计算知识蒸馏的损失
#         # TODO
#
#         # 计算综合的损失
#         outputs = (target_logits,) + outputs[1:]
#         outputs = (loss, ) + outputs
#
#         return outputs

# 貌似没有用到
class BartSummaryGenerationHead(nn.Module):
    def __init__(self, config, shared):
        super(BartSummaryGenerationHead, self).__init__(config)
        self.predict_dense = nn.Linear(config.d_model, config.vocab_size)
        self.shared = shared
        self.init_linear_weight(self.predict_dense)

    def forward(self, hiddens):
        logits = self.predict_dense(hiddens)
        return logits

    def init_linear_weight(self, m):
        """使用shared参数初始化线性层m参数  在摘要中即将最后预测层的参数赋值为embedding的权重 """

        # 声明变量final_logits_bias 并赋予初值；和直接=赋值不同的是 使用register_buffer使得变量在optim.step中不会进行更新
        self.register_buffer("final_logits_bias", torch.zeros(1, self.shared.num_embeddings))
        m.bias = self.final_logtis_bias
        m.weight = self.shared.weight


class BartForOnlySummaryGeneration(PretrainedBartModel):
    """在bart模型上只进行xsum or cnn/dm的 fine tune, 没有任何其他的组件
        在预测层同样需要训练学习 不是直接和embedding 层进行共享。
        只是赋初值的时候使用embedding 的权重
    """

    def __init__(self, config, args):
        super(BartForOnlySummaryGeneration, self).__init__(config, args)
        base_model = BartModel(config)
        self.model = base_model
        self.label_smoothing = args.label_smoothing
        # # register_buffer表示的是权重不进行更新, num_embedding 表示词表的大小
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        labels=None,
        use_cache=False,
        **unused
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."
            model = BartForConditionalGeneration.from_pretrained('bart-large')
            input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]
            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)
            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = unused.pop("lm_labels")

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )

        decoder_outputs = outputs["decoder_output"][0]

        # 每次进行预测的时候都是使用已经训练好的embedding层的参数
        # 未经正则的logits
        lm_logits = F.linear(decoder_outputs, self.model.shared.weight, bias=self.final_logits_bias)     # 直接使用embedding的权重初始化最后的预测层的
        sentence_nums = labels.size(0)    # 使用句子数进行label smooth
        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        if labels is not None:
            loss_fct = LabelSmoothedCrossEntropyCriterion(
                label_smoothing=self.label_smoothing, padding_idx=self.config.pad_token_id
            )
            loss, nll_loss = loss_fct(lm_logits, labels, normalization=sentence_nums)
            outputs = (loss, nll_loss) + outputs

        return outputs

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        if not past[1]:
            encoder_outputs, decoder_cached_states = past, None
        else:
            encoder_outputs, decoder_cached_states = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_logits_for_generation(self, logits, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.shared)  # make it on the fly

