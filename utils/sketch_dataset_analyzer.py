import json
import os
import random
import sys

from datetime import datetime as dt
from enum import Enum, unique


def match_sketch_withShapeNet(taxonomy_file_path):
    rendering_path = '/mnt/Entertainment/图像数据集/20211219手绘数据集/%s/%s/rendering/%02d.png'
    # Load all taxonomies of the dataset
    with open(taxonomy_file_path, encoding='utf-8') as file:
        dataset_taxonomy = json.loads(file.read())
        sketch_taxonomies = []
        # Load data for each category
        for taxonomy in dataset_taxonomy:
            sketch_samples = []
            taxonomy_folder_name = taxonomy['taxonomy_id']
            print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s]' % (dt.now(), taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
            samples = taxonomy['train'] + taxonomy['test'] + taxonomy['val']
            for sample_idx, sample_name in enumerate(samples):
                # Get file list of rendering images
                img_file_path = rendering_path % (taxonomy_folder_name, sample_name, 0)
                if not os.path.exists(img_file_path):
                    print('[WARN] %s Ignore sample %s/%s since imgs do not exist.' %  (dt.now(), taxonomy_folder_name, sample_name))
                else:
                    sketch_samples.append(sample_name)

            sample_num = len(sketch_samples)
            sample_train, sample_test = int(sample_num*0.7), int(sample_num*0.2)
            sample_val = sample_num-sample_train-sample_test
            sketch_taxonomy = {
                "taxonomy_id": taxonomy['taxonomy_id'],
                "taxonomy_name": taxonomy['taxonomy_name'],
                "baseline": taxonomy['baseline'],
                "test": sketch_samples[0:sample_test],
                "train": sketch_samples[sample_test:sample_test+sample_train],
                "val": sketch_samples[sample_test+sample_train:]
            }

            sketch_taxonomies.append(sketch_taxonomy)

    return sketch_taxonomies


'''def get_files_of_taxonomy(self, taxonomy_folder_name, samples):
    files_of_taxonomy = []

    for sample_idx, sample_name in enumerate(samples):
        # Get file path of volumes
        volume_file_path = self.volume_path_template % (taxonomy_folder_name, sample_name)
        if not os.path.exists(volume_file_path):
            print('[WARN] %s Ignore sample %s/%s since volume file not exists.' %
                  (dt.now(), taxonomy_folder_name, sample_name))
            continue

        # Get file list of rendering images
        img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, 0)
        if not os.path.exists(img_file_path):
            print('[WARN] %s Ignore sample %s/%s since imgs file not exists.' %
                  (dt.now(), taxonomy_folder_name, sample_name))
            continue
        img_folder = os.path.dirname(img_file_path)
        total_views = len(os.listdir(img_folder))
        rendering_image_indexes = range(total_views)
        rendering_images_file_path = []
        for image_idx in rendering_image_indexes:
            img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
            if not os.path.exists(img_file_path):
                continue

            rendering_images_file_path.append(img_file_path)

        if len(rendering_images_file_path) == 0:
            print('[WARN] %s Ignore sample %s/%s since image files not exists.' %
                  (dt.now(), taxonomy_folder_name, sample_name))
            continue

        # Append to the list of rendering images
        files_of_taxonomy.append({
            'taxonomy_name': taxonomy_folder_name,
            'sample_name': sample_name,
            'rendering_images': rendering_images_file_path,
            'volume': volume_file_path,
        })

        # Report the progress of reading dataset
        # if sample_idx % 500 == 499 or sample_idx == n_samples - 1:
        #     print('[INFO] %s Collecting %d of %d' % (dt.now(), sample_idx + 1, n_samples))

    return files_of_taxonomy'''

if __name__ == '__main__':
    out_path = '../datasets/20211219.json'
    taxonomy_file_path = '../datasets/ShapeNet/ShapeNet.json'
    # Load all taxonomies of the dataset
    with open(out_path, mode='w', encoding='utf-8') as file:
        sketch_dataset = match_sketch_withShapeNet(taxonomy_file_path)
        json.dump(sketch_dataset, file, indent=4)