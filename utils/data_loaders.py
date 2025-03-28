# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
import os
import random
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset
from config import cfg

from datetime import datetime as dt
from enum import Enum, unique

import utils.binvox_rw


@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


# //////////////////////////////// = End of DatasetType Class Definition = ///////////////////////////////// #
class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, dataset_type, file_list, n_views_rendering, transforms=None):
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.transforms = transforms
        self.n_views_rendering = n_views_rendering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_id, taxonomy_name, sample_name, rendering_images, volume, points = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return taxonomy_id, taxonomy_name, sample_name, rendering_images, volume, points

    def set_n_views_rendering(self, n_views_rendering):
        self.n_views_rendering = n_views_rendering

    def get_datum(self, idx):
        taxonomy_id = self.file_list[idx]['taxonomy_id']
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        volume_path = self.file_list[idx]['volume']
        point_path = self.file_list[idx]['point']

        # Get data of rendering images
        if self.dataset_type == DatasetType.TRAIN:
            selected_rendering_image_paths = [
                # rendering_image_paths[i]
                rendering_image_paths[0]  # 修改为只使用0号图片训练
                for i in random.sample(range(len(rendering_image_paths)), self.n_views_rendering)
            ]
        else:
            selected_rendering_image_paths = [rendering_image_paths[i] for i in range(self.n_views_rendering)]

        rendering_images = []
        for image_path in selected_rendering_image_paths:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if len(rendering_image.shape) < 3:
                print('[FATAL] %s It seems that there is something wrong with the image file %s' %(dt.now(), image_path))
                sys.exit(2)

            rendering_images.append(rendering_image)

        # Get data of volume
        _, suffix = os.path.splitext(volume_path)
        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)

        with open(file=point_path, mode='r') as f:
            lines = f.readlines()
            points = []
            for idx, line in enumerate(lines):
                if idx < cfg.CONST.NPOINT:
                    xy = line.strip().split(',')
                    points.append([int(xy[0]), int(xy[1])])
                else:
                    break
        points = np.array(points).astype(np.float32)

        return taxonomy_id, taxonomy_name, sample_name, np.asarray(rendering_images), volume, points


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #
class ShapeNetDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.rendering_image_path_template = cfg.DATASETS.SHAPENET.RENDERING_PATH
        self.volume_path_template = cfg.DATASETS.SHAPENET.VOXEL_PATH
        self.point_path = cfg.DATASETS.SHAPENET.POINT_PATH

        # Load all taxonomies of the dataset
        filepath = os.path.join(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, cfg.CONST.CATEGORY+'.json')
        with open(filepath, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_id = taxonomy['taxonomy_id']
            taxonomy_name = taxonomy['taxonomy_name']
            print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s]' % (dt.now(), taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['val']

            files.extend(self.get_files_of_taxonomy(taxonomy_id, taxonomy_name, samples))

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (dt.now(), len(files)))
        return ShapeNetDataset(dataset_type, files, n_views_rendering, transforms)

    def get_files_of_taxonomy(self, taxonomy_id, taxonomy_name, samples):
        files_of_taxonomy = []

        for sample_idx, sample_name in enumerate(samples):
            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_id, sample_name)
            if not os.path.exists(volume_file_path):
                print('[WARN] %s Ignore sample %s/%s since volume file not exists.' % (dt.now(), taxonomy_id, sample_name))
                continue

            # Get file list of rendering images
            img_file_path = self.rendering_image_path_template % (taxonomy_id, sample_name, 0)
            if not os.path.exists(img_file_path):
                print('[WARN] %s Ignore sample %s/%s since imgs file not exists.' % (dt.now(), taxonomy_id, sample_name))
                continue
            img_folder = os.path.dirname(img_file_path)
            total_views = len(os.listdir(img_folder))
            rendering_image_indexes = range(total_views)
            rendering_images_file_path = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_id, sample_name, image_idx)
                if not os.path.exists(img_file_path):
                    continue
                rendering_images_file_path.append(img_file_path)

            if len(rendering_images_file_path) == 0:
                print('[WARN] %s Ignore sample %s/%s since image files not exists.' %
                      (dt.now(), taxonomy_id, sample_name))
                continue

            point_path = self.point_path % (taxonomy_id, sample_name)
            if not os.path.exists(point_path):
                print('[WARN] %s Ignore sample %s/%s since point file not exists.' % (dt.now(), taxonomy_id, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_id' : taxonomy_id,
                'taxonomy_name': taxonomy_name,
                'sample_name': sample_name,
                'rendering_images': rendering_images_file_path,
                'volume': volume_file_path,
                'point': point_path,
            })

        return files_of_taxonomy
# /////////////////////////////// = End of ShapeNetDataLoader Class Definition = /////////////////////////////// #

class Pix3dDataset(torch.utils.data.dataset.Dataset):
    """Pix3D class used for PyTorch DataLoader"""
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_id, taxonomy_name, sample_name, rendering_images, volume, points = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return taxonomy_id, taxonomy_name, sample_name, rendering_images, volume, points

    def get_datum(self, idx):
        taxonomy_id = self.file_list[idx]['taxonomy_id']
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_path = self.file_list[idx]['rendering_image']
        volume_path = self.file_list[idx]['volume']
        point_path = self.file_list[idx]['point']

        # Get data of rendering images
        rendering_image = cv2.imread(rendering_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if len(rendering_image.shape) < 3:
            print('[WARN] %s It seems the image file %s is grayscale.' % (dt.now(), rendering_image_path))
            rendering_image = np.stack((rendering_image, ) * 3, -1)

        # Get data of volume
        _, suffix = os.path.splitext(volume_path)
        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)

        # Get data of point
        with open(file=point_path, mode='r') as f:
            lines = f.readlines()
            points = []
            for idx, line in enumerate(lines):
                if idx < cfg.CONST.NPOINT:
                    xy = line.strip().split(',')
                    points.append([int(xy[0]), int(xy[1])])
                else:
                    break
        points = np.array(points).astype(np.float32)

        return taxonomy_id, taxonomy_name, sample_name, np.asarray([rendering_image]), volume, points
# //////////////////////////////// = End of Pix3dDataset Class Definition = ///////////////////////////////// #


class Pix3dDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.volume_path_template = cfg.DATASETS.PIX3D.VOXEL_PATH
        self.rendering_image_path_template = cfg.DATASETS.PIX3D.RENDERING_PATH
        self.point_path = cfg.DATASETS.PIX3D.POINT_PATH

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.PIX3D.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_id = taxonomy['taxonomy_id']
            taxonomy_name = taxonomy['taxonomy_name']
            print('[INFO] %s Collecting files of Taxonomy[Name=%s]' % (dt.now(), taxonomy_name))

            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['val']

            files.extend(self.get_files_of_taxonomy(taxonomy_id, taxonomy_name, samples))

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (dt.now(), len(files)))
        return Pix3dDataset(files, transforms)

    def get_files_of_taxonomy(self, taxonomy_id, taxonomy_name, samples):
        files_of_taxonomy = []
        for sample_idx, sample_name in enumerate(samples):
            # Get rendering image
            rendering_image_file_path = self.rendering_image_path_template % (taxonomy_id, sample_name)

            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_id, sample_name)
            if not os.path.exists(volume_file_path):
                print('[WARN] %s Ignore sample %s/%s since volume file not exists.' %(dt.now(), taxonomy_id, sample_name))
                continue

            point_path = self.point_path % (taxonomy_id, sample_name)
            if not os.path.exists(point_path):
                print('[WARN] %s Ignore sample %s/%s since point file not exists.' % (dt.now(), taxonomy_id, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_id': taxonomy_id,
                'taxonomy_name': taxonomy_name,
                'sample_name': sample_name,
                'rendering_image': rendering_image_file_path,
                'volume': volume_file_path,
                'point': point_path
            })

        return files_of_taxonomy


# /////////////////////////////// = End of Pix3dDataLoader Class Definition = /////////////////////////////// #


# /////////////////////////////// Begin of ModelNetDataset Definition /////////////////////////////// #
class ModelNetDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_id, taxonomy_name, sample_name, rendering_images, volume, points = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return taxonomy_id, taxonomy_name, sample_name, rendering_images, volume, points

    def get_datum(self, idx):
        taxonomy_id = self.file_list[idx]['taxonomy_id']
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_path = self.file_list[idx]['rendering_image']
        volume_path = self.file_list[idx]['volume']
        point_path = self.file_list[idx]['point']

        # Get data of rendering images
        rendering_image = cv2.imread(rendering_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if len(rendering_image.shape) < 3:
            print('[WARN] %s It seems the image file %s is grayscale.' % (dt.now(), rendering_image_path))
            rendering_image = np.stack((rendering_image, ) * 3, -1)

        # Get data of volume
        _, suffix = os.path.splitext(volume_path)
        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)

        # Get data of point
        with open(file=point_path, mode='r') as f:
            lines = f.readlines()
            points = []
            for idx, line in enumerate(lines):
                if idx < cfg.CONST.NPOINT:
                    xy = line.strip().split(',')
                    points.append([int(xy[0]), int(xy[1])])
                else:
                    break
        points = np.array(points).astype(np.float32)

        return taxonomy_id, taxonomy_name, sample_name, np.asarray([rendering_image]), volume, points
# //////////////////////////////// End of ModelNetDataset Class Definition ///////////////////////////////// #

# //////////////////////////////// Begin of ModelNetDataLoader Class Definition ///////////////////////////////// #
class ModelNetDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.volume_path_template = cfg.DATASETS.MODELNET.VOXEL_PATH
        self.rendering_image_path_template = cfg.DATASETS.MODELNET.RENDERING_PATH
        self.point_path = cfg.DATASETS.MODELNET.POINT_PATH

        # Load all taxonomies of the dataset
        filepath = os.path.join(cfg.DATASETS.MODELNET.TAXONOMY_FILE_PATH, cfg.CONST.CATEGORY + '.json')
        with open(filepath, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_name = taxonomy['taxonomy_name']
            taxonomy_id = taxonomy['taxonomy_id']
            print('[INFO] %s Collecting files of Taxonomy[Name=%s]' % (dt.now(), taxonomy_name))

            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['val']

            files.extend(self.get_files_of_taxonomy(taxonomy_id, taxonomy_name, samples))

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (dt.now(), len(files)))
        return ModelNetDataset(files, transforms)

    def get_files_of_taxonomy(self, taxonomy_id, taxonomy_name, samples):
        files_of_taxonomy = []
        for sample_idx, sample_name in enumerate(samples):
            # Get rendering image
            rendering_image_file_path = self.rendering_image_path_template % (taxonomy_id, sample_name)

            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_id, sample_name)
            if not os.path.exists(volume_file_path):
                print('[WARN] %s Ignore sample %s/%s since volume file not exists.' %(dt.now(), taxonomy_id, sample_name))
                continue

            point_path = self.point_path % (taxonomy_id, sample_name)
            if not os.path.exists(point_path):
                print('[WARN] %s Ignore sample %s/%s since point file not exists.' % (dt.now(), taxonomy_id, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_id': taxonomy_id,
                'taxonomy_name': taxonomy_name,
                'sample_name': sample_name,
                'rendering_image': rendering_image_file_path,
                'volume': volume_file_path,
                'point': point_path
            })

        return files_of_taxonomy
# /////////////////////////////// End of ModelNetDataLoader Class Definition /////////////////////////////// #


DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader,
    'ModelNet': ModelNetDataLoader,
    'Pix3D': Pix3dDataLoader
}

