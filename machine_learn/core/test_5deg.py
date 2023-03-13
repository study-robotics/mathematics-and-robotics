# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import numpy as np
import logging
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from models.encoder import Encoder
from models.decoder_5deg import Decoder
from utils.average_meter import AverageMeter




def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST),
                                                batch_size=1,
                                                num_workers=cfg.CONST.NUM_WORKER,
                                                pin_memory=True,
                                                shuffle=False)


    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()

        logging.info('Loading weights from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = []
    encoder_losses = AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()

    for sample_idx, (volumes, ground_truth_cspace) in enumerate(test_data_loader):
        #taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        #sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            volumes = utils.helpers.var_or_cuda(volumes)
            ground_truth_cspace = utils.helpers.var_or_cuda(ground_truth_cspace)

            # Test the encoder, decoder
            volume_features = encoder(volumes)
            generated_cspace = decoder(volume_features)

            generated_cspace = torch.mean(generated_cspace, dim=1)

            # add binarization
            #generated_cspace = utils.helpers.get_binarization_cspace(generated_cspace)

            encoder_loss = bce_loss(generated_cspace, ground_truth_cspace) * 10

            # Append loss and accuracy to average metrics
            encoder_losses.update(encoder_loss.item())

            # caculate IoU
            # https://qiita.com/4Ui_iUrz1/items/4c0efd9c50e344c66665
            sample_iou = []
            for th in cfg.TEST.CSPACE_THRESH:
                _cspace = torch.ge(generated_cspace, th).float()
                intersection = torch.sum(_cspace.mul(ground_truth_cspace)).float()
                union = torch.sum(torch.ge(_cspace.add(ground_truth_cspace), 1)).float()
                sample_iou.append((intersection / union).item())
            test_iou.append(sample_iou)

            # Append generated volumes to TensorBoard
            view_sample = 0 # TensorBoardにアップロードする画像の枚数 # 更に学習に時間がかかるので，基本は0でOK
            save_sample = 3 # ローカルの「./output」に保存する画像の枚数
            if test_writer and sample_idx < view_sample:
                # Cspace Visualization
                after_generated_probability_cspace = utils.helpers.get_probability_cspace_view(generated_cspace.cpu().numpy())
                test_writer.add_image('Model%02d/ProbabilityReconstructed' % sample_idx, after_generated_probability_cspace, epoch_idx)
                after_generated_cspace = utils.helpers.get_cspace_views(generated_cspace.cpu().numpy())
                test_writer.add_image('Model%02d/Reconstructed' % sample_idx, after_generated_cspace, epoch_idx)
                after_ground_truth_cspace = utils.helpers.get_cspace_views(ground_truth_cspace.cpu().numpy())
                test_writer.add_image('Model%02d/GroundTruth' % sample_idx, after_ground_truth_cspace, epoch_idx)

            save_epoch = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
            if (epoch_idx in save_epoch) and sample_idx < save_sample:
                utils.helpers.save_generated_cspace(sample_idx, epoch_idx, generated_cspace.cpu().numpy())
                utils.helpers.save_probability_cspace(sample_idx, epoch_idx, generated_cspace.cpu().numpy())            
                utils.helpers.save_ground_truth_cspace(sample_idx, epoch_idx, ground_truth_cspace.cpu().numpy())            

            #  test用C-spaceの生成結果を保存したい場合は，下記のコメントアウトをはずす
            """
            utils.helpers.save_generated_cspace(sample_idx, epoch_idx, generated_cspace.cpu().numpy())
            utils.helpers.save_probability_cspace(sample_idx, epoch_idx, generated_cspace.cpu().numpy())            
            utils.helpers.save_ground_truth_cspace(sample_idx, epoch_idx, ground_truth_cspace.cpu().numpy())
            """    

            # Print sample loss and IoU
            logging.info('Test[%d/%d] EDLoss = %.4f  IoU = %s' %
                         (sample_idx + 1, n_samples, encoder_loss.item(),
                           ['%.4f' % si for si in sample_iou]))
    
    mean_iou = np.mean(test_iou, axis=0)
    max_iou = np.max(mean_iou)
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')
    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    print("mean_iou:", np.mean(mean_iou))
    print("mean_loss:", encoder_losses.avg)

    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('EncoderDecoder/IoU', max_iou, epoch_idx)

    return max_iou
