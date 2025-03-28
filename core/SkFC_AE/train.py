# -*- coding: utf-8 -*-
import os

import torch
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.SkFC_AE.val import val_net

from models.modules.ffm import FeatureFusionModule
from models.modules.local_image_encoder import ResEncoder
from models.modules.global_points_encoder import PointNetEncoder
from models.modules.mlp import Mlp, Adaptation
from models.modules.fcm import FeatureComplementModule
from models.modules.decoder import Decoder
from models.modules.global_image_encoder import FLattenSwinTransformer






def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Resize(IMG_SIZE),
        #utils.data_transforms.Binarize(threshold=0.2),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Resize(IMG_SIZE),
        # utils.data_transforms.Binarize(threshold=0.2),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.TRAIN.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=1,
                                                  num_workers=1,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up networks
    # vit = Vit()

    lie = ResEncoder(cfg)
    gpe = PointNetEncoder(channel=2)
    gie = FLattenSwinTransformer(num_classes=1024)
    FFM = FeatureFusionModule(cfg)
    mlp = Adaptation(1024)
    decoder = Decoder(cfg)
    FCM = FeatureComplementModule(cfg)

    print('[DEBUG] %s Parameters in LIE: %d.' % (dt.now(), utils.network_utils.count_parameters(lie)))
    print('[DEBUG] %s Parameters in GIE: %d.' % (dt.now(), utils.network_utils.count_parameters(gie)))
    print('[DEBUG] %s Parameters in GPE: %d.' % (dt.now(), utils.network_utils.count_parameters(gpe)))
    print('[DEBUG] %s Parameters in FFM: %d.' % (dt.now(), utils.network_utils.count_parameters(FFM)))
    print('[DEBUG] %s Parameters in FCM: %d.' % (dt.now(), utils.network_utils.count_parameters(FCM)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in Mlp: %d.' % (dt.now(), utils.network_utils.count_parameters(mlp)))

    total_param = utils.network_utils.count_parameters(lie) + utils.network_utils.count_parameters(mlp) + \
                     utils.network_utils.count_parameters(gpe) + \
                     utils.network_utils.count_parameters(gie) + utils.network_utils.count_parameters(decoder) + \
                     utils.network_utils.count_parameters(FFM) + utils.network_utils.count_parameters(FCM)

    print('[DEBUG] %s Parameters in Total: %d.' % (dt.now(), total_param))


    # Initialize weights of networks
    lie.apply(utils.network_utils.init_weights)
    gpe.apply(utils.network_utils.init_weights)
    gie.apply(utils.network_utils.init_weights)
    FFM.apply(utils.network_utils.init_weights)
    decoder.apply(utils.network_utils.init_weights)
    FCM.apply(utils.network_utils.init_weights)
    mlp.apply(utils.network_utils.init_weights)


    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':

        lie_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, lie.parameters()), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        gpe_solver = torch.optim.Adam(gpe.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        FFM_solver = torch.optim.Adam(FFM.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        gie_solver = torch.optim.Adam(gie.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        FCM_solver = torch.optim.Adam(FCM.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        mlp_solver = torch.optim.Adam(mlp.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)

    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    # encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_solver, cfg.TRAIN.DECAY_LR_EVERY_EPOCH, cfg.TRAIN.DECAY_LR_RATING)
    # attn_lr_scheduler = torch.optim.lr_scheduler.StepLR(attn_solver, cfg.TRAIN.DECAY_LR_EVERY_EPOCH, cfg.TRAIN.DECAY_LR_RATING)
    # vit_lr_scheduler = torch.optim.lr_scheduler.StepLR(vit_solver, cfg.TRAIN.DECAY_LR_EVERY_EPOCH, cfg.TRAIN.DECAY_LR_RATING)
    # ea_lr_scheduler = torch.optim.lr_scheduler.StepLR(ea_solver, cfg.TRAIN.DECAY_LR_EVERY_EPOCH, cfg.TRAIN.DECAY_LR_RATING)
    # vae_parameter_lr_scheduler = torch.optim.lr_scheduler.StepLR(get_vae_parameter_solver, cfg.TRAIN.DECAY_LR_EVERY_EPOCH, cfg.TRAIN.DECAY_LR_RATING)
    # pointNetEncoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(pointNetEncoder_solver, cfg.TRAIN.DECAY_LR_EVERY_EPOCH, cfg.TRAIN.DECAY_LR_RATING)
    # fusion_lr_scheduler = torch.optim.lr_scheduler.StepLR(fusion_solver, cfg.TRAIN.DECAY_LR_EVERY_EPOCH, cfg.TRAIN.DECAY_LR_RATING)
    # decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(decoder_solver, cfg.TRAIN.DECAY_LR_EVERY_EPOCH, cfg.TRAIN.DECAY_LR_RATING)
    # refiner_lr_scheduler = torch.optim.lr_scheduler.StepLR(refiner_solver, cfg.TRAIN.DECAY_LR_EVERY_EPOCH, cfg.TRAIN.DECAY_LR_RATING)

    if torch.cuda.is_available():
        local_image_encoder = torch.nn.DataParallel(lie).cuda()
        global_points_encoder = torch.nn.DataParallel(gpe).cuda()
        FFM = torch.nn.DataParallel(FFM).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        global_image_encoder = torch.nn.DataParallel(gie).cuda()
        FCM = torch.nn.DataParallel(FCM).cuda()
        mlp = torch.nn.DataParallel(mlp).cuda()

    init_epoch = 0
    best_iou = -1
    best_epoch = -1

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat().replace(":" , "_"))
    ckpt_dir = output_dir % 'checkpoints'
    log_dir = output_dir % 'logs'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'val'))

    # Training
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        epoch_start_time = time()  # Tick/tock

        # Batch average meterics
        # data_time = utils.network_utils.AverageMeter()
        total_losses = utils.network_utils.AverageMeter()

        # switch models to training mode
        local_image_encoder.train()
        global_points_encoder.train()
        FFM.train()
        decoder.train()
        global_image_encoder.train()
        FCM.train()
        mlp.train()

        batch_end_time = time()
        with tqdm(train_data_loader, desc='train network') as train_data_loader:
            for batch_idx, (taxonomy_ids, taxonomy_names, sample_names, rendering_images, ground_truth_volumes, image_points) in enumerate(train_data_loader):
                # data_time.update(time() - batch_end_time)  # Measure data time

                # Get data from data loader
                rendering_images = utils.network_utils.var_or_cuda(rendering_images)
                ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)
                image_points = utils.network_utils.var_or_cuda(image_points)

                image_features = local_image_encoder(rendering_images)

                global_features = global_image_encoder(rendering_images.squeeze(1))

                point_features = global_points_encoder(image_points)

                image_features, global_features, point_features = mlp(image_features, global_features, point_features)

                image_features = image_features.reshape(-1, 16, 64)
                global_features = global_features.reshape(-1, 16, 64)
                point_features = point_features.reshape(-1, 16, 64)

                common_voxel = FCM(global_features, point_features, s_mask=None)

                image_global_voxel, points_voxel = FFM(image_features, global_features, point_features)

                gen_volume = decoder(common_voxel, image_global_voxel)
                gen_volume1 = gen_volume.squeeze(1)

                total_loss = utils.network_utils.MSE_MSFCEL(gen_volume1, ground_truth_volumes)

                # Gradient decent
                local_image_encoder.zero_grad()
                global_points_encoder.zero_grad()
                global_image_encoder.zero_grad()
                FCM.zero_grad()
                FFM.zero_grad()
                decoder.zero_grad()
                mlp.zero_grad()

                total_loss.backward()

                lie_solver.step()
                gpe_solver.step()
                FFM_solver.step()
                FCM_solver.step()
                gie_solver.step()
                decoder_solver.step()
                mlp_solver.step()

              

            '''# Adjust learning rate
            encoder_lr_scheduler.step()
            pointNetEncoder_lr_scheduler.step()
            transformer_lr_scheduler.step()
            decoder_lr_scheduler.step()
            refiner_lr_scheduler.step()'''


        # # Adjust learning rate
        # encoder_lr_scheduler.step()
        # pointNetEncoder_lr_scheduler.step()
        # attn_lr_scheduler.step()
        # vit_lr_scheduler.step()
        # ea_lr_scheduler.step()
        # vae_parameter_lr_scheduler.step()
        # fusion_lr_scheduler.step()
        # decoder_lr_scheduler.step()
        # refiner_lr_scheduler.step()

        # Append loss to average metrics
        # encoder_losses.update(encoder_loss.item())
        total_losses.update(total_loss.item())

        # Tick / tock
        epoch_end_time = time()
        print('[Train loss] %s Epoch [%d/%d] EpochTime = %.3f (s) total_loss = %.4f'
              % (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, total_losses.avg))
        # Append epoch loss to TensorBoard
        # train_writer.add_scalar('Train/encoder_loss', total_losses.avg, epoch_idx + 1)

        # Validate the training models
        iou = val_net(cfg=cfg, epoch_idx=epoch_idx+1, val_writer=val_writer, data_loader=val_data_loader,
                      local_image_encoder=local_image_encoder, global_points_encoder=global_points_encoder, FFM=FFM, FCM=FCM, decoder=decoder,
                      global_image_encoder=global_image_encoder, mlp=mlp)

        '''# Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(cfg=cfg, file_path=os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth' % (epoch_idx + 1)), epoch_idx=epoch_idx + 1, encoder=encoder, encoder_solver=encoder_solver,
                                                 decoder=decoder,decoder_solver=decoder_solver, refiner=refiner,refiner_solver=refiner_solver, best_iou=best_iou, best_epoch=best_epoch)'''
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            best_iou = iou
            best_epoch = epoch_idx + 1
            ckpt_file = os.path.join(ckpt_dir, 'best-ckpt.pth')
            if os.path.exists(ckpt_file):
                os.remove(ckpt_file)
            utils.network_utils.save_checkpoints(cfg=cfg, file_path=ckpt_file, epoch_idx=epoch_idx + 1, best_epoch=best_epoch, best_iou=best_iou,
                                                 lie=local_image_encoder, lie_solver=lie_solver,
                                                 gpe=global_points_encoder, gpe_solver=gpe_solver,
                                                 gie=global_image_encoder, gie_solver=gie_solver,
                                                 ffm=FFM, ffm_solver=FFM_solver,
                                                 fcm = FCM, fcm_solver =FCM_solver,
                                                 decoder=decoder, decoder_solver=decoder_solver,
                                                 mlp=mlp, mlp_solver=mlp_solver)
        if iou > best_iou:
            best_iou = iou
            best_epoch = epoch_idx + 1
        print("epoch: %s, iou: %s, best_iou: %s, best_epoch: %s" % (epoch_idx+1, iou, best_iou, best_epoch))
        # Close SummaryWriter for TensorBoard
        train_writer.close()
        val_writer.close()
