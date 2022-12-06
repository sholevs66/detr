import argparse
import datetime
import json
import random
import time
from pathlib import Path
import logging

import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

from main import get_args_parser
from einops import rearrange, repeat

import onnx
from onnxsim import simplify


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., out_bias = False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = out_bias),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    '''
    # org
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    '''
    def forward(self, query, key, value, attn_mask, key_padding_mask):

        q = self.to_q(query) # [1,1800, 256]
        k = self.to_k(key) # [1,1800, 256]
        v = self.to_v(value) # [1,1800, 256]
        #qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = 8), (q, k, v)) # [1, 8, 1800, 32], [1, 8, 1800, 32], [1, 8, 1800, 32]
        
        #q_ = rearrange(q, 'b n (h d) -> b h n d', h=8)
        #k_ = rearrange(k, 'b n (h d) -> b h n d', h=8)
        #v_ = rearrange(v, 'b n (h d) -> b h n d', h=8)


        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def replace_attention(model):
    
    # encoder 
    for child in model.transformer.encoder.layers.children():
        for name, layer in child.named_children():
            if isinstance(layer, nn.MultiheadAttention):
                in_proj_weight = layer.in_proj_weight
                in_proj_bias = layer.in_proj_bias
                out_proj_weight = layer.out_proj.weight
                out_proj_bias = layer.out_proj.bias

                dim = int(in_proj_bias.shape[0]/3)
                dim_head = int(dim / args.nheads)
                out_bias = bool(out_proj_bias.detach().cpu().numpy().sum())
                attention_vit = Attention(dim=dim, dim_head=dim_head, dropout=0.1, out_bias=out_bias)
                attention_vit.to_qkv.weight = in_proj_weight

                attention_vit.to_q.weight = nn.Parameter(in_proj_weight[0:256])
                attention_vit.to_k.weight = nn.Parameter(in_proj_weight[256:512])
                attention_vit.to_v.weight = nn.Parameter(in_proj_weight[512:])
                attention_vit.to_q.bias = nn.Parameter(in_proj_bias[0:256])
                attention_vit.to_k.bias = nn.Parameter(in_proj_bias[256:512])
                attention_vit.to_v.bias = nn.Parameter(in_proj_bias[512:])
                attention_vit.to_out[0].weight = out_proj_weight
                if out_bias:
                    attention_vit.to_out[0].bias = out_proj_bias
                child.add_module(name, attention_vit)

    for child in model.transformer.decoder.layers.children():
        for name, layer in child.named_children():
            if isinstance(layer, nn.MultiheadAttention):
                in_proj_weight = layer.in_proj_weight
                in_proj_bias = layer.in_proj_bias
                out_proj_weight = layer.out_proj.weight
                out_proj_bias = layer.out_proj.bias

                dim = int(in_proj_bias.shape[0]/3)
                dim_head = int(dim / args.nheads)
                out_bias = bool(out_proj_bias.detach().cpu().numpy().sum())
                attention_vit = Attention(dim=dim, dim_head=dim_head, dropout=0.1, out_bias=out_bias)
                attention_vit.to_qkv.weight = in_proj_weight
                attention_vit.to_q.weight = nn.Parameter(in_proj_weight[0:256])
                attention_vit.to_k.weight = nn.Parameter(in_proj_weight[256:512])
                attention_vit.to_v.weight = nn.Parameter(in_proj_weight[512:])
                attention_vit.to_q.bias = nn.Parameter(in_proj_bias[0:256])
                attention_vit.to_k.bias = nn.Parameter(in_proj_bias[256:512])
                attention_vit.to_v.bias = nn.Parameter(in_proj_bias[512:])
                attention_vit.to_out[0].weight = out_proj_weight
                if out_bias:
                    attention_vit.to_out[0].bias = out_proj_bias
                child.add_module(name, attention_vit)


def main(args):

    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    if not args.resume:
        print('No loaded model!')

    if args.resume:
        a = torch.load(args.resume, map_location='cpu')
        if 'model' in a:
            model.load_state_dict(a['model'])
        else:
            model.load_state_dict(a)
            

    replace_attention(model)

    # to onnx
    model.eval()
    imgs = torch.zeros(1,3,800,800, dtype=torch.float32).to(device)
    outputs = model(imgs)
    torch.onnx.export(model, imgs, args.out_name + '.onnx', input_names=['test_input'], output_names=['logits', 'boxes'], training=torch.onnx.TrainingMode.PRESERVE, opset_version=11)
    print('model saved at: ', args.out_name + '.onnx')

    # if also simplify
    if args.no_simplify:
        model_onnx = onnx.load(args.out_name + '.onnx')
        model_simp, check = simplify(model_onnx)
        onnx.save(model_simp, args.out_name + '_simplify.onnx')
        print('model simplified saved at: ', args.out_name + '_simplify.onnx')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(epilog='Example: CUDA_VISIBLE_DEVICES=0 python export_to_onnx.py --batch_first --enc_bn --dec_bn --no_aux_loss --backbone resnet50 --no_return_intermediate_dec --out_name my_detr_model.onnx', parents=[get_args_parser()])
    parser.add_argument('--out_name', default='detr', type=str, help="Name for the onnx output")
    parser.add_argument('--no_simplify', action='store_false')
    args = parser.parse_args()
    
    main(args)



