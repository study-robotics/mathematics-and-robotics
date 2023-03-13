# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os
import logging
import random
import torch
from torch._C import dtype
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.opt_test import test_net
from models.encoder import Encoder
from models.decoder import Decoder
from utils.average_meter import AverageMeter



def l1_loss(model):
    l1 = torch.tensor(0., requires_grad=True)
    for w in model.parameters():
        l1 = l1 + torch.norm(w, 1)
    return l1

def orth_loss(model):
    '''
    直交正則化
    '''
    # 損失の計算
    # loss = ...
    #reg = torch.tensor(1e-6).float().cuda() 
    orth_loss = torch.zeros(1).cuda()
    for name, param in model.named_parameters():
        if 'bias' not in name:
            param_flat = param.view(param.shape[0], -1)
            sym = torch.mm(param_flat, torch.t(param_flat))
            sym -= torch.eye(param_flat.shape[0]).cuda()
            #orth_loss = orth_loss + (reg * sym.abs().sum())
            orth_loss = orth_loss + (sym.abs().sum())
    return orth_loss


def train_net(cfg, alpha=0, beta=0, encoder_weight_decay=0.0, decoder_weight_decay=0.0):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.TRAIN),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    """
    for data,label in train_data_loader:
        break
    print(data)
    print(label)
    """
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.VAL),
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.NUM_WORKER,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    logging.debug('Parameters in Encoder: %d.' % (utils.helpers.count_parameters(encoder)))
    logging.debug('Parameters in Decoder: %d.' % (utils.helpers.count_parameters(decoder)))
    
    # Initialize weights of networks
    encoder.apply(utils.helpers.init_weights)
    decoder.apply(utils.helpers.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS,
                                          weight_decay=encoder_weight_decay)
        decoder_solver = torch.optim.Adam(decoder.parameters(),
                                          lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS,
                                          weight_decay=decoder_weight_decay)

    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM,
                                         weight_decay=cfg.REGULARIZATION)
        decoder_solver = torch.optim.SGD(decoder.parameters(),
                                         lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM,
                                         weight_decay=cfg.REGULARIZATION)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    minimize_loss = -1
    best_epoch = -1

    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    cfg.DIR.LOGS = output_dir % 'logs'
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHS):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        encoder_losses = AverageMeter()

        # switch models to training mode
        encoder.train()
        decoder.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (volumes, ground_truth_cspace) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)

            # Get data from data loader
            volumes = utils.helpers.var_or_cuda(volumes)
            ground_truth_cspace = utils.helpers.var_or_cuda(ground_truth_cspace)


            # Train the encoder, decoder
            volume_features = encoder(volumes)
            generated_cspace = decoder(volume_features)
        
            generated_cspace = torch.mean(generated_cspace, dim=1)

            encoder_loss = bce_loss(generated_cspace, ground_truth_cspace) * 10
            encoder_orth_loss = orth_loss(encoder)
            decoder_orth_loss = orth_loss(decoder)
            encoder_l1_loss = l1_loss(encoder)
            decoder_l1_loss = l1_loss(decoder)
            #alpha = 0 # 1e-4 orth
            #beta = 0
            encoder_loss = encoder_loss \
                           + alpha * encoder_orth_loss + beta * decoder_orth_loss 

            # Gradient decent
            encoder.zero_grad()
            decoder.zero_grad()


            encoder_loss.backward()

            encoder_solver.step()
            decoder_solver.step()

            # Append loss to average metrics
            encoder_losses.update(encoder_loss.item())

            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)

            """
            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info(
                '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f' %
                (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, batch_idx + 1, n_batches, batch_time.val, data_time.val,
                 encoder_loss.item()))
            """

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        """
        logging.info('[Epoch %d/%d] EpochTime = %.3f (s) EDLoss = %.4f' %
                     (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, epoch_end_time - epoch_start_time, encoder_losses.avg))
        """

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, val_data_loader, val_writer, encoder, decoder)

        """
        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0 or iou > best_iou:
            file_name = 'checkpoint-epoch-%03d.pth' % (epoch_idx + 1)
            if iou > best_iou:
                best_iou = iou
                best_epoch = epoch_idx
                file_name = 'checkpoint-best.pth'

            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            if not os.path.exists(cfg.DIR.CHECKPOINTS):
                os.makedirs(cfg.DIR.CHECKPOINTS)

            checkpoint = {
                'epoch_idx': epoch_idx,
                'best_iou': best_iou,
                'best_epoch': best_epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
            }

            torch.save(checkpoint, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)
        """
        if iou > best_iou:
            best_iou = iou
            
    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()
    print("best_iou", best_iou)
    return best_iou