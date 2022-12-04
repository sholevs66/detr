# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

from util.load_model import load_pretrained_weights

from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--lr_drop', default=300, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--enc_bn', action='store_true')
    parser.add_argument('--dec_bn', action='store_true')
    parser.add_argument('--batch_first', action='store_true')

    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--freeze_enc', action='store_true')
    parser.add_argument('--freeze_dec', action='store_true')
    parser.add_argument('--freeze_query', action='store_true')
    parser.add_argument('--freeze_input_proj', action='store_true') # 2048 -> 256 channels in R50->transformer
    parser.add_argument('--freeze_bbox_embed', action='store_true')
    parser.add_argument('--freeze_class_embed', action='store_true')
    parser.add_argument('--freeze_all', action='store_true')
    parser.add_argument('--mix_precision', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--no_return_intermediate_dec', dest='return_intermediate_dec', action='store_false',
                        help="For exporting to onnx this needs to be false")
                    
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    if args.freeze_enc:
        for p in model.transformer.encoder.parameters():
            p.requires_grad = False

    if args.freeze_dec:
        for p in model.transformer.decoder.parameters():
            p.requires_grad = False

    if args.freeze_query:
        for p in model.query_embed.parameters():
            p.requires_grad = False

    if args.freeze_input_proj:
        for p in model.input_proj.parameters():
            p.requires_grad = False

    if args.freeze_bbox_embed:
        for p in model.bbox_embed.parameters():
            p.requires_grad = False

    if args.freeze_class_embed:
        for p in model.class_embed.parameters():
            p.requires_grad = False

    if args.freeze_all:
        for p in model.parameters():
            p.requires_grad = False


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        #model_without_ddp.load_state_dict(checkpoint['model'])  # original

        # omer try to match org model with BN model
        load_pretrained_weights(model_without_ddp, checkpoint)

        '''
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        '''

    '''
    # Sometimes we want to freeze all model and then turn on grads for certain specific layers
    model_without_ddp.transformer.encoder.layers[5].linear1.bias.requires_grad = True
    model_without_ddp.transformer.encoder.layers[5].linear1.weight.requires_grad = True
    model_without_ddp.transformer.encoder.layers[5].linear2.bias.requires_grad = True
    model_without_ddp.transformer.encoder.layers[5].linear2.weight.requires_grad = True
    model_without_ddp.transformer.encoder.layers[5].norm1.bias.requires_grad = True
    model_without_ddp.transformer.encoder.layers[5].norm1.weight.requires_grad = True
    model_without_ddp.transformer.encoder.layers[5].norm2.bias.requires_grad = True
    model_without_ddp.transformer.encoder.layers[5].norm2.weight.requires_grad = True
    model_without_ddp.transformer.encoder.layers[5].norm3.bias.requires_grad = True
    model_without_ddp.transformer.encoder.layers[5].norm3.weight.requires_grad = True
    '''
    #model_without_ddp.transformer.encoder.layers[0].norm1.reset_parameters()
    #model_without_ddp.transformer.encoder.layers[0].norm2.reset_parameters()
    #model_without_ddp.transformer.encoder.layers[0].norm3.reset_parameters()

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    # logging
    if args.output_dir:
        logging.basicConfig(filename=os.path.join(args.output_dir, 'info.log') , level=logging.INFO)
        logging.info('  resume: %s', args.resume)
        logging.info('  lr: %f', args.lr)
        logging.info('  lr_backbone: %f', args.lr_backbone)
        logging.info('  batch_size: %d', args.batch_size)
        logging.info('  weight_decay: %f', args.weight_decay)
        logging.info('  epochs: %d', args.epochs)
        logging.info('  lr_drop: %d', args.lr_drop)
        logging.info('  enc_layers: %d', args.enc_layers)
        logging.info('  dec_layers: %d', args.dec_layers)
        logging.info('  dim_feedforward: %d', args.dim_feedforward)
        logging.info('  hidden_dim: %d', args.hidden_dim)
        logging.info('  dropout: %f', args.dropout)
        logging.info('  pre-norm: %s', args.pre_norm)
        logging.info('  enc_bn: %s', args.enc_bn)
        logging.info('  dec_bn: %s', args.dec_bn)
        logging.info('  batch_first: %s', args.batch_first)
        logging.info('  mix_precision: %s', args.mix_precision)
        logging.info('  freeze_backbone: %s', args.freeze_backbone)
        logging.info('  freeze_enc: %s', args.freeze_enc)
        logging.info('  freeze_dec: %s', args.freeze_dec)


    print("Start training")
    best_val_ap = 0  # init val ap tracker
    writer = SummaryWriter(log_dir=args.output_dir)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, args.mix_precision)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 50 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        
        # train & validation stats to tensorboard
        a=list(coco_evaluator.coco_eval.values())[0]
        writer.add_scalar('mAP', a.stats[0], epoch)
        writer.add_scalars("class_error", {'train':train_stats['class_error'], 'test':test_stats['class_error']}, epoch)
        writer.add_scalars("loss", {'train':train_stats['loss'], 'test':test_stats['loss']}, epoch)
        writer.add_scalars("loss_ce", {'train':train_stats['loss_ce'], 'test':test_stats['loss_ce']}, epoch)
        writer.add_scalars("loss_bbox", {'train':train_stats['loss_bbox'], 'test':test_stats['loss_bbox']}, epoch)
        writer.add_scalars("loss_giou", {'train':train_stats['loss_giou'], 'test':test_stats['loss_giou']}, epoch)

        # save best val AP mode
        if a.stats[0] > best_val_ap:
            best_val_ap = a.stats[0]
            torch.save(model_without_ddp.state_dict(), os.path.join(args.output_dir, 'model_best.pth'))

        if args.output_dir:
            logging.info('  best val AP: %f', best_val_ap)


        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    writer.flush()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
