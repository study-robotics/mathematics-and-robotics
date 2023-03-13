# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                        = edict()
__C.DATASETS.VOX_CSPACE             = edict()
__C.DATASETS.VOX_CSPACE.JSON_PATH   = "/workspace/Vox2Cspace/datasets/vox_cspace.json"
__C.DATASETS.VOX_CSPACE.VOXEL_PATH  = "/workspace/Vox2Cspace/datasets/voxels/%s.binvox"
__C.DATASETS.VOX_CSPACE.CSPACE_PATH = "/workspace/Vox2Cspace/datasets/Cspaces/%s.npy"
#__C.DATASETS.VOX_CSPACE.VOXEL_PATH  = "/workspace/Vox2Cspace/datasets/test/voxels/%s.binvox"
#__C.DATASETS.VOX_CSPACE.CSPACE_PATH = "/workspace/Vox2Cspace/datasets/test/Cspaces/test_10deg/%s.npy"

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'VoxCspace'
__C.DATASET.TEST_DATASET                    = 'VoxCspace'

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.BATCH_SIZE                        = 32        # default 64
__C.CONST.NUM_WORKER                        = 0         # default 4: number of data workers

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './output'
__C.DIR.RANDOM_BG_PATH                      = '/home/hzxie/Datasets/SUN2012/JPEGImages'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False
__C.NETWORK.USE_REFINER                     = False
__C.NETWORK.USE_MERGER                      = True

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_EPOCHS                        = 50
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .4
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
__C.REGULARIZATION                          = 0      # 1e-2 l2 reguration of encoder
__C.TRAIN.EPOCH_START_USE_REFINER           = 0
__C.TRAIN.EPOCH_START_USE_MERGER            = 0
__C.TRAIN.ENCODER_LEARNING_RATE             = 1e-3
__C.TRAIN.DECODER_LEARNING_RATE             = 1e-3
__C.TRAIN.REFINER_LEARNING_RATE             = 1e-3
__C.TRAIN.MERGER_LEARNING_RATE              = 1e-4
__C.TRAIN.ENCODER_LR_MILESTONES             = [20]
__C.TRAIN.DECODER_LR_MILESTONES             = [20]
__C.TRAIN.REFINER_LR_MILESTONES             = [20]
__C.TRAIN.MERGER_LR_MILESTONES              = [20]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 10            # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING          = False

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.CSPACE_THRESH                       = [.2, .3, .4, .5] # ボクセル化するしきい値
