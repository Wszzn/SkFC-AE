# -*- coding: utf-8 -*-
import json
import numpy as np
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

from models.modules.ffm import FeatureFusionModule
from models.modules.local_image_encoder import ResEncoder
from models.modules.global_points_encoder import PointNetEncoder
from models.modules.mlp import Mlp, Adaptation
from models.modules.fcm import FeatureComplementModule
from models.modules.decoder import Decoder
from models.modules.global_image_encoder import FLattenSwinTransformer


def test_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat().replace(":" , "_"))

    # Load taxonomies of dataset
    taxonomies = []
    # with open(cfg.DATASETS[cfg.DATASET.TRAIN_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
    filepath = os.path.join(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, cfg.CONST.CATEGORY + '.json')
    with open(filepath, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Resize(IMG_SIZE),
        # utils.data_transforms.Binarize(threshold=0.2),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                   batch_size=1,
                                                   num_workers=1,
                                                   pin_memory=True,
                                                   shuffle=False)

    # Set up networks
    lie = ResEncoder(cfg)
    gpe = PointNetEncoder(channel=2)
    gie = FLattenSwinTransformer(num_classes=1024)
    FFM = FeatureFusionModule(cfg)
    mlp = Adaptation(1024)
    decoder = Decoder(cfg)
    FCM = FeatureComplementModule(cfg)


    if torch.cuda.is_available():
        local_image_encoder = torch.nn.DataParallel(lie).cuda()
        global_points_encoder = torch.nn.DataParallel(gpe).cuda()
        FFM = torch.nn.DataParallel(FFM).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        global_image_encoder = torch.nn.DataParallel(gie).cuda()
        FCM = torch.nn.DataParallel(FCM).cuda()
        mlp = torch.nn.DataParallel(mlp).cuda()

    print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    epoch_idx = checkpoint['epoch_idx']

    # Load state dictionaries for all models matching the save_checkpoints parameters
    local_image_encoder.load_state_dict(checkpoint['lie_state_dict'])
    global_points_encoder.load_state_dict(checkpoint['gpe_state_dict'])
    global_image_encoder.load_state_dict(checkpoint['gie_state_dict'])
    FFM.load_state_dict(checkpoint['ffm_state_dict'])
    FCM.load_state_dict(checkpoint['fcm_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    mlp.load_state_dict(checkpoint['mlp_state_dict'])


    # Validating loop
    test_iou = dict()

    # Switch models to evaluation mode
    local_image_encoder.eval()
    global_points_encoder.eval()
    FFM.eval()
    FCM.eval()
    decoder.eval()
    global_image_encoder.eval()
    mlp.eval()

    with tqdm(test_data_loader, desc='test network') as test_data_loader:
        for sample_idx, (taxonomy_ids, taxonomy_names, sample_names, rendering_images, ground_truth_volumes, image_points) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_ids[0]
            sample_name = sample_names[0]
            ground_truth_volume = ground_truth_volumes[0]

            with torch.no_grad():
                # Get data from data loader
                rendering_images = utils.network_utils.var_or_cuda(rendering_images)
                image_points = utils.network_utils.var_or_cuda(image_points)
                ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

                image_features = local_image_encoder(rendering_images)

                global_features = global_image_encoder(rendering_images.squeeze(1))

                point_features = global_points_encoder(image_points)

                image_features, global_features, point_features = mlp(image_features, global_features, point_features)

                common_voxel = FCM(global_features, point_features, s_mask=None)

                image_global_voxel, points_voxel = FFM(image_features, global_features, point_features)

                gen_volume = decoder(common_voxel, image_global_voxel)

                gen_volume = gen_volume.squeeze()

                # IoU per sample
                sample_iou = []
                for th in cfg.TEST.VOXEL_THRESH:
                    _volume = torch.ge(gen_volume, th).float()
                    intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                    union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                    sample_iou.append((intersection / union).item())

                # IoU per taxonomy
                if taxonomy_id not in test_iou:
                    test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
                test_iou[taxonomy_id]['n_samples'] += 1
                test_iou[taxonomy_id]['iou'].append(sample_iou)

                '''# 保存测试生成的图像
                if output_dir:
                    img_dir = output_dir % 'images'
                    # Volume Visualization
                    gv = generated_volume.cpu().numpy()
                    rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir, 'best-epoch-'+str(epoch_idx)), sample_name, max(sample_iou), epoch_idx)
                    gtv = ground_truth_volume.cpu().numpy()
                    rendering_views = utils.binvox_visualization.get_volume_views(gtv, os.path.join(img_dir, 'best-epoch-'+str(epoch_idx)), sample_name+'-gt', max(sample_iou), epoch_idx)'''

                # 保存测试生成的3D模型
                if output_dir:
                    binvox_dir = output_dir % 'binvoxs'
                    binvox_file = os.path.join(binvox_dir, taxonomy_id, '%s_%s_%.4f.binvox' % (sample_name, 'S2V', max(sample_iou)))
                    #binvox_file = os.path.join(binvox_dir, taxonomy_id, sample_name, '%s.binvox' % sample_name)
                    if not os.path.exists(os.path.dirname(binvox_file)):
                        os.makedirs(os.path.dirname(binvox_file))
                    gv = gen_volume.cpu().numpy()
                    gv = (gv.squeeze() > cfg.TEST.VOXEL_THRESH[0]).astype(np.int32)  # change numpy datatype to bool
                    with open(binvox_file, 'wb') as f:
                        #voxel_model = utils.binvox_rw.Voxels(data=gv, dims=gv.shape, translate=[0, 0, 0], scale=1, axis_order='xyz')
                        voxel_model = utils.binvox_rw.Voxels(data=gv, dims=gv.shape, translate=[0, 0, 0], scale=1, axis_order='xzy')
                        voxel_model.write(f)

                '''# Print sample loss and IoU
                print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
                      (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
                       refiner_loss.item(), ['%.4f' % si for si in sample_iou]))'''

    # Output testing results
    mean_iou = []
    n_samples = 0  # len(test_data_loader)
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
        n_samples += test_iou[taxonomy_id]['n_samples']
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS (best epoch %d)============================' % epoch_idx)
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%-7d' % test_iou[taxonomy_id]['n_samples'], end='\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall\t\t%-7d' % n_samples, end='\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    max_iou = np.max(mean_iou)
    return max_iou
