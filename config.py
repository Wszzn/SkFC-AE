# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                                = edict()

__C.DATASETS.SHAPENET                       = edict()

# E.G. TAXONOMY_FILE_PATH
# __C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH  = './datasets/ShapeNet/'
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH  = '/PATH/TO/YOUR/TAXONOMY_FILE_PATH'

# E.G. POINT_PATH
# __C.DATASETS.SHAPENET.POINT_PATH = './datasets/ShapeNet/ShapeNet_contour/%s/%s/00.txt'
__C.DATASETS.SHAPENET.POINT_PATH = '/PATH/TO/YOUR/POINT_FILE_PATH'

# E.G. SKETCH_PATH
# __C.DATASETS.SHAPENET.RENDERING_PATH = './datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.RENDERING_PATH = '/PATH/TO/YOUR/SKETCH_FILE_PATH'

# E.G. VOXEL_GT_PATH
# __C.DATASETS.SHAPENET.VOXEL_PATH = './datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'
__C.DATASETS.SHAPENET.VOXEL_PATH = '/PATH/TO/YOUR/VOXEL_GT_PATH'

__C.DATASETS.MODELNET                          = edict()
__C.DATASETS.MODELNET.TAXONOMY_FILE_PATH       = '/PATH/TO/YOUR/TAXONOMY_FILE_PATH'
__C.DATASETS.MODELNET.RENDERING_PATH           = '/PATH/TO/YOUR/SKETCH_FILE_PATH'
__C.DATASETS.MODELNET.VOXEL_PATH               = '/PATH/TO/YOUR/VOXEL_GT_PATH'
__C.DATASETS.MODELNET.POINT_PATH = '/PATH/TO/YOUR/POINT_FILE_PATH'
#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]

__C.DATASET.STD                             = [0.5, 0.5, 0.5]

__C.DATASET.TRAIN_DATASET                   = 'ShapeNet'
__C.DATASET.TEST_DATASET                    = 'ShapeNet'

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.N_VOX                             = 32
__C.CONST.BATCH_SIZE                        = 16
__C.CONST.N_VIEWS_RENDERING                 = 1
__C.CONST.CROP_IMG_W                        = 128
__C.CONST.CROP_IMG_H                        = 128

__C.CONST.NPOINT = 256
__C.CONST.D_MODEL = 64
__C.CONST.D_MODEL1 = 64
__C.CONST.N_HEAD = 4
__C.CONST.DROP_PROB = 0.2
__C.CONST.N_LAYERS = 4
__C.CONST.N_LAYERS1 = 4
__C.CONST.WEIGHTS = r'/path/to/best-ckpt.pth'
__C.CONST.CATEGORY = 'chair'
# Directories
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './output'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2

__C.NETWORK.TCONV_USE_BIAS                  = False

__C.NETWORK.USE_REFINER                     = True
__C.NETWORK.USE_MERGER                      = False

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_WORKER                        = 0           # number of data workers
__C.TRAIN.NUM_EPOCHES                       = 200
__C.TRAIN.NUM_EPOCHES_AUTOENCODER           = 0
__C.TRAIN.NUM_EPOCHES_REGRESS               = 0
#__C.TRAIN.NUM_EPOCHES_FINETUNE              = 250
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .4
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER           = 0
__C.TRAIN.EPOCH_START_USE_MERGER            = 0
__C.TRAIN.LEARNING_RATE                     = 1e-3
__C.TRAIN.ENCODER_LEARNING_RATE             = 5e-5
__C.TRAIN.DECODER_LEARNING_RATE             = 5e-5
__C.TRAIN.REFINER_LEARNING_RATE             = 5e-5
__C.TRAIN.VAE_PARAMS_LEARNING_RATE              = 5e-5
__C.TRAIN.LR_MILESTONES             = [150]
__C.TRAIN.ENCODER_LR_MILESTONES             = [150]
__C.TRAIN.DECODER_LR_MILESTONES             = [150]
__C.TRAIN.REFINER_LR_MILESTONES             = [150]
__C.TRAIN.MERGER_LR_MILESTONES              = [150]
__C.TRAIN.DECAY_LR_EVERY_EPOCH  = 100
__C.TRAIN.DECAY_LR_RATING = 0.5
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
__C.TEST.VOXEL_THRESH                       = [.5]
