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
# __C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = 'F:\\backup\\backup\\sketch2vox\datasets'
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = './datasets'
#__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH  = './datasets/ShapeNet.json'
# __C.DATASETS.SHAPENET.POINT_PATH = './datasets/ShapeNet/512_txt_sort/%s/%s/00.txt'
__C.DATASETS.SHAPENET.POINT_PATH = './datasets/ShapeNet/ShapeNet_contour/%s/%s/00.txt'
__C.DATASETS.SHAPENET.RENDERING_PATH = './datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
# __C.DATASETS.SHAPENET.RENDERING_PATH = '/home/file_Wdisk/dataset/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH = './datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'

__C.DATASETS.PIX3D                          = edict()
__C.DATASETS.PIX3D.TAXONOMY_FILE_PATH       = './datasets/pix3d.json'
__C.DATASETS.PIX3D.RENDERING_PATH           = './datasets/pix3d/image/%s/%s/model.png'
__C.DATASETS.PIX3D.VOXEL_PATH               = './datasets/pix3d/binvox/%s/%s/model.binvox'
__C.DATASETS.PIX3D.POINT_PATH = './datasets/pix3d/pointSet/%s/%s/model.txt'

__C.DATASETS.MODELNET                          = edict()
__C.DATASETS.MODELNET.TAXONOMY_FILE_PATH       = './datasets/ModelNet'
#__C.DATASETS.MODELNET.RENDERING_PATH           = './datasets/ModelNet10/image/%s/%s/model.png'
__C.DATASETS.MODELNET.RENDERING_PATH           = './datasets/ModelNet/ModelNet/ModelNet10_contour/%s/%s/5.png'
__C.DATASETS.MODELNET.VOXEL_PATH               = './datasets/ModelNet/ModelNet/binvox/%s/%s.binvox'
#__C.DATASETS.MODELNET.POINT_PATH = './datasets/ModelNet10/pointSet/%s/%s/model.txt'
__C.DATASETS.MODELNET.POINT_PATH = './datasets/ModelNet10/ModelNet10_pointset/point/%s/%s/5.txt'
#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
#均值 channel,height,width
# 将两个参数都设置为0.5并与transforms.ToTensor()一起使用可以使将数据强制缩放到[-1,1]区间上。（标准化只能保证大部分数据在0附近——3σ原则）
__C.DATASET.STD                             = [0.5, 0.5, 0.5]#方差 channel,height,width
# __C.DATASET.TRAIN_DATASET                   = 'ShapeNet'
# __C.DATASET.TEST_DATASET                    = 'ShapeNet'
#__C.DATASET.TRAIN_DATASET                   = 'Pix3D'
#__C.DATASET.TEST_DATASET                    = 'Pix3D'
__C.DATASET.TRAIN_DATASET                   = 'ModelNet'
__C.DATASET.TEST_DATASET                    = 'ModelNet'

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
#__C.CONST.Z_LENGTH                          = 512     # the length of Z value
#__C.CONST.FEATURE_DIM = 64  # encoder和decoder中间的特征向量长度
__C.CONST.NPOINT = 256  # 使用pointNet结构前,取关键点数量
__C.CONST.D_MODEL = 64  # 使用self-attention后每个特征向量的长度
__C.CONST.D_MODEL1 = 64
__C.CONST.N_HEAD = 8  # 多头自注意机制
__C.CONST.DROP_PROB = 0.1 # 防止过拟合,以drop_prob概率随机丢弃部分分支
__C.CONST.N_LAYERS = 4  #融合模块由n_layers层多头自注意机制构成 DEFAULT = 4
__C.CONST.N_LAYERS1 = 4
__C.CONST.WEIGHTS = './output/checkpoints/2024-07-01T04_47_07.636131/best-ckpt.pth'
__C.CONST.CATEGORY = 'chair'
# Directories
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './output'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
# Relu的输入值为负的时候，输出始终为0，其一阶导数也始终为0，这样会导致神经元不能更新参数，也就是神经元不学习了，这种现象叫做“Dead Neuron”。
# 为了解决Relu函数这个缺点，在Relu函数的负半区间引入一个泄露（Leaky）值，所以称为Leaky Relu函数
__C.NETWORK.TCONV_USE_BIAS                  = False
# 就像一般的Dense （y = W*x + b）一样，卷积层同样有一个偏移量b，输出为Conv(w,x) + b
# use_bias参数的作用就是决定卷积层输出是否有b。
# 设为True时有，False没有
__C.NETWORK.USE_REFINER                     = True
__C.NETWORK.USE_MERGER                      = False

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_WORKER                        = 10             # number of data workers
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
