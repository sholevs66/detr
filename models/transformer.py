# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from res_mlp_files import mlp_mixer


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, enc_bn=False, dec_bn=False, enc_resmlp=False):
        super().__init__()
        #import ipdb; ipdb.set_trace()

        if enc_resmlp == False:
            if enc_bn==False:
                encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                        dropout, activation, normalize_before)
            else:
                encoder_layer = TransformerEncoderLayer_BN(d_model, nhead, dim_feedforward,
                                                        dropout, activation, normalize_before)

            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        else:
            self.encoder = mlp_mixer.resmlp_12_400_bn_relu_detr()

        if dec_bn==False:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
        else:
            decoder_layer = TransformerDecoderLayer_BN(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            decoder_norm = nn.BatchNorm1d(d_model) # check if this is the problem in all BN DETR
            #decoder_norm = nn.LayerNorm(d_model)

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.enc_resmlp = enc_resmlp

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        # src = [batch, 256, H/32, W/32]
        # mask - [batch, H/32 * W/32] - Bool padding for batch attention - pad to max in each batch
        # query_embed - [100, 256] - learned priors for decoder

        bs, c, h, w = src.shape
        src = src.flatten(2).permute(0, 2, 1)   # [batch, query, channel] = [2, 400, 256] - for resmlp
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src) # for resmplp
        memory = memory.permute(1,0,2) # [batch, query, channel] -> [query, batch, channel]
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # memory_key_padding_mask - [batch, query] - mask for padded pixels - colate_fn to match batch size or to get to 640x640

        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                if isinstance(self.norm, nn.LayerNorm):
                    intermediate.append(self.norm(output))
                else:
                    # if BN
                    #output = output.permute(1, 2, 0) # [qeury, batch, channel]=[100,2,256] -> [batch, channel, query]
                    #output = self.norm(output)
                    #output = output.permute(2, 0, 1) # [batch, channel, query] -> [qeury, batch, channel]
                    #intermediate.append(output)
                    intermediate.append(self.norm(output.permute(1, 2, 0)).permute(2, 0, 1))

        #import ipdb; ipdb.set_trace()
        if self.norm is not None:
            if isinstance(self.norm, nn.LayerNorm):
                output = self.norm(output)
            else:
                # If BN
                output = output.permute(1, 2, 0) # [qeury, batch, channel]=[100,2,256] -> [batch, channel, query]
                output = self.norm(output)
                output = output.permute(2, 0, 1) # [batch, channel, query] -> [qeury, batch, channel]
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)  # [#queries, batch, 256] -> [1, #queries, batch, 256]


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(src, pos)   # [qeury, batch, channel] = [768, 2, 256]
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # # [qeury, batch, channel] = [768, 2, 256]
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoderLayer_BN(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if normalize_before:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
            self.norm3 = nn.BatchNorm1d(dim_feedforward)
        else:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(dim_feedforward)
            self.norm3 = nn.BatchNorm1d(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos) # [qeury, batch, channel] = [768, 2, 256]
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2) # [qeury, batch, channel] = [768, 2, 256]
        #import ipdb; ipdb.set_trace()
        src = src.permute(1, 2, 0) # [qeury, batch, channel] -> [batch, channel, query]
        src = self.norm1(src) # compute statistics for each channel over the batch,query -> mean=[channel,1] & var=[channel,1]
        src = src.permute(2, 0, 1) # [batch, channel, query] -> [qeury, batch, channel]

        src2 = self.linear1(src)  # [qeury, batch, dim_feedforward] = [768, 2, 2048]
        src2 = src2.permute(1, 2, 0) # [qeury, batch, dim_feedforward] -> [batch, dim_feedforward, query]
        src2 = self.norm2(src2)
        src2 = src2.permute(2, 0, 1) # [batch, dim_feedforward, query] -> [qeury, batch, dim_feedforward]
        src2 = self.linear2(self.dropout(self.activation(src2))) # [qeury, batch, channel] = [768, 2, 256]

        src = src + self.dropout2(src2)
        src = src.permute(1, 2, 0) # [qeury, batch, channel] -> [batch, channel, query]
        src = self.norm3(src)
        src = src.permute(2, 0, 1) # [batch, channel, query] -> [qeury, batch, channel]
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src.permute(1, 2, 0)).permute(2, 0, 1)

        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src.permute(1, 2, 0)).permute(2, 0, 1)
        
        #src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = self.linear1(src2)
        src2 = src2.permute(1, 2, 0)
        src2 = self.norm3(src2)
        src2 = src2.permute(2, 0, 1)
        src2 = self.linear2(self.dropout(self.activation(src2)))

        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        #import ipdb; ipdb.set_trace()
        q = k = self.with_pos_embed(tgt, query_pos)     # [#queries, #batch, 256]
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)   # normalizes over the last dim - the 256 + multiply & addition by learned params
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2) # [#queries, #batch, 256]
        tgt = self.norm2(tgt) # [#queries, #batch, 256], normalizes over the last dim - the 256 + multiply & addition by learned params
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt)))) # [#queries, #batch, 256]
        tgt = tgt + self.dropout3(tgt2) # [#queries, #batch, 256]
        tgt = self.norm3(tgt) # [#queries, #batch, 256], normalizes over the last dim - the 256 + multiply & addition by learned params
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        #import ipdb; ipdb.set_trace()
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoderLayer_BN(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if normalize_before:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
            self.norm3 = nn.BatchNorm1d(d_model)
            self.norm4 = nn.BatchNorm1d(dim_feedforward)
        else:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
            self.norm3 = nn.BatchNorm1d(dim_feedforward)
            self.norm4 = nn.BatchNorm1d(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)     # tgt=q=k=[#queries=100, #batch, 256]
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        #tgt = self.norm1(tgt)   # normalizes over the last dim - the 256 + multiply & addition by learned params

        tgt = tgt.permute(1, 2, 0) # [qeury, batch, channel] -> [batch, channel, query]
        tgt = self.norm1(tgt)
        tgt = tgt.permute(2, 0, 1) # [batch, channel, query] -> [qeury, batch, channel]

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2) # [#queries, #batch, 256]
        
        #tgt = self.norm2(tgt) # [#queries, #batch, 256], normalizes over the last dim - the 256 + multiply & addition by learned params
        tgt = tgt.permute(1, 2, 0) # [qeury, batch, channel] -> [batch, channel, query]
        tgt = self.norm2(tgt)
        tgt = tgt.permute(2, 0, 1) # [batch, channel, query] -> [qeury, batch, channel]

        #tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt)))) # [#queries, #batch, 256]

        tgt2 = self.linear1(tgt)  # [qeury, batch, dim_feedforward] = [768, 2, 2048]
        tgt2 = tgt2.permute(1, 2, 0) # [qeury, batch, dim_feedforward] -> [batch, dim_feedforward, query]
        tgt2 = self.norm3(tgt2)
        tgt2 = tgt2.permute(2, 0, 1) # [batch, dim_feedforward, query] -> [qeury, batch, dim_feedforward]
        tgt2 = self.linear2(self.dropout(self.activation(tgt2))) # [qeury, batch, channel] = [768, 2, 256]

        tgt = tgt + self.dropout3(tgt2) # [#queries, #batch, 256]

        #tgt = self.norm3(tgt) # [#queries, #batch, 256], normalizes over the last dim - the 256 + multiply & addition by learned params
        tgt = tgt.permute(1, 2, 0)
        tgt = self.norm4(tgt)
        tgt = tgt.permute(2, 0, 1)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt.permute(1, 2, 0)).permute(2, 0, 1)

        #tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt2 = self.linear1(tgt2)
        tgt2 = tgt2.permute(1, 2, 0)
        tgt2 = self.norm4(tgt2)
        tgt2 = tgt2.permute(2, 0, 1)
        tgt2 = self.linear2(self.dropout(self.activation(tgt2)))

        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        #import ipdb; ipdb.set_trace()
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        enc_bn=args.enc_bn,
        dec_bn=args.dec_bn,
        enc_resmlp=args.enc_resmlp
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
