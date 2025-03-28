import json
import os
import random
import sys
from datetime import datetime as dt
from enum import Enum, unique
import cv2
import numpy as np
from config import cfg

def get_image_pointset(rendering_image):
    """
    Input:
        rendering_image: image data, [height, width, channel]
        npoint: number of sample points
    Return:
        all_points: sampled pointcloud index, [npoint, 2]
    """
    # search_value = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    img_points = rendering_image > 0
    img_points = np.where(img_points[...])
    img_points = np.concatenate([np.expand_dims(coord, axis=1) for coord in img_points], axis=1)
    return img_points

def match_sketch_withShapeNet(taxonomy_file_path):
    rendering_path = '/mnt/Entertainment/图像数据集/20211219手绘数据集/%s/%s/rendering/%02d.png'
    # Load all taxonomies of the dataset
    with open(taxonomy_file_path, encoding='utf-8') as file:
        dataset_taxonomy = json.loads(file.read())
        sketch_taxonomies = []
        # Load data for each category
        del_num = 0  # 删除数据个数
        total_num = 0  # 删除前总数据量
        for taxonomy in dataset_taxonomy:
            sketch_samples = []
            taxonomy_folder_name = taxonomy['taxonomy_id']
            print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s]' % (dt.now(), taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
            samples = taxonomy['train'] + taxonomy['test'] + taxonomy['val']
            for sample_idx, sample_name in enumerate(samples):
                # Get file list of rendering images
                img_file_path = rendering_path % (taxonomy_folder_name, sample_name, 0)
                img = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
                img = 255 - img
                img_point = get_image_pointset(img)
                total_num += 1
                if img_point.shape[0] <= cfg.CONST.NPOINT:
                    del_num += 1
                else:
                    sketch_samples.append(sample_name)
                '''if img_point.shape[0] <= cfg.CONST.NPOINT:
                    print('[WARN] %s Ignore sample %s/%s since imgs do not exist.' % (dt.now(), taxonomy_folder_name, sample_name))
                else:
                    sketch_samples.append(sample_name)'''

            # 按 7:2:1的比例分配数据集, 作为train_dataset,test_dataset,val_dataset
            sample_num = len(sketch_samples)
            sample_train, sample_test = int(sample_num*0.7), int(sample_num*0.2)
            sample_val = sample_num-sample_train-sample_test
            sketch_taxonomy = {
                "taxonomy_id": taxonomy['taxonomy_id'],
                "taxonomy_name": taxonomy['taxonomy_name'],
                "test": sketch_samples[0:sample_test],
                "train": sketch_samples[sample_test:sample_test+sample_train],
                "val": sketch_samples[sample_test+sample_train:]
            }

            sketch_taxonomies.append(sketch_taxonomy)
    print("删除数据个数: %d/%d. " % (del_num, total_num))
    return sketch_taxonomies


if __name__ == '__main__':
    out_path = '../datasets/sofa_20220330.json'
    taxonomy_file_path = '../datasets/20211219.json'
    # Load all taxonomies of the dataset
    with open(out_path, mode='w', encoding='utf-8') as file:
        sketch_dataset = match_sketch_withShapeNet(taxonomy_file_path)
        json.dump(sketch_dataset, file, indent=4)