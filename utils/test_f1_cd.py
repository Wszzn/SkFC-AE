import os
import glob

import numpy as np

from external import binvox_rw
from scipy.spatial.distance import cdist


def chamfer_distance(pred_voxel, gt_voxel):
    pred_points = np.argwhere(pred_voxel)
    gt_points = np.argwhere(gt_voxel)

    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0

    pred_to_gt_dist = cdist(pred_points, gt_points, 'euclidean')
    gt_to_pred_dist = cdist(gt_points, pred_points, 'euclidean')

    min_pred_dist = np.min(pred_to_gt_dist, axis=1)
    min_gt_dist = np.min(gt_to_pred_dist, axis=1)

    cd = np.mean(min_pred_dist) + np.mean(min_gt_dist)
    return cd


def f1_score(pred_voxel, gt_voxel):
    # 计算 true positives, false positives 和 false negatives
    tp = np.sum(np.logical_and(pred_voxel == 1, gt_voxel == 1))  # 预测和GT都是1的体素
    fp = np.sum(np.logical_and(pred_voxel == 1, gt_voxel == 0))  # 预测为1，GT为0的体素
    fn = np.sum(np.logical_and(pred_voxel == 0, gt_voxel == 1))  # 预测为0，GT为1的体素

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def load_binvox(filename):
    with open(filename, 'rb') as f:
        voxels = binvox_rw.read_as_3d_array(f)
    voxels = voxels.data.astype(np.float32)
    return voxels


def evaluate(pred_binvox, gt_binvox):
    pred_voxel = load_binvox(pred_binvox).transpose(0, 2, 1)
    # pred_voxel = load_binvox(pred_binvox)
    gt_voxel = load_binvox(gt_binvox)

    # compute CD
    cd = chamfer_distance(pred_voxel, gt_voxel)

    # compute F1
    f1 = f1_score(pred_voxel, gt_voxel)

    return f1, cd

def evaluate_multiple_files(pred_dir, gt_dir):

    pred_files = glob.glob(os.path.join(pred_dir, '**\**\*.binvox'), recursive=True)

    gt_files = glob.glob(os.path.join(gt_dir, '**\*.binvox'), recursive=True)

    results = []

    for pred_file in pred_files:
        file_name = pred_file.split('\\')[5]
        file_name = os.path.basename(file_name).split('_S2V')[0]

        gt_file = next((gt for gt in gt_files if file_name in gt), None)

        if gt_file:
            f1, cd = evaluate(pred_file, gt_file)
            results.append((file_name, f1, cd))
            print(f'Processed: {file_name} | F1 Score: {f1} | Chamfer Distance: {cd}')
        else:
            print(f'No GT found for: {file_name}')

    return results


pred_dir = 'F:\Datasets\deepsdf_shapenet\deepsdf'
gt_dir = 'F:\Datasets\ShapeNetVox32'

# 批量处理并获取结果
results = evaluate_multiple_files(pred_dir, gt_dir)

average_f1 = []
average_cd = []

for result in results:
    file_name, f1, cd = result
    if not np.isnan(f1):
        average_f1.append(f1)
    else:
        print(f"Warning: F1 Score for {file_name} is nan, skipping this result.")
        average_f1.append(0)

    average_cd.append(cd)
    print(f'File: {file_name} | F1 Score: {f1} | Chamfer Distance: {cd}')

# 计算平均值时，确保没有 nan 值
if len(average_f1) > 0:
    print("Average F1:", np.mean(average_f1), "Average CD:", np.mean(average_cd))
else:
    print("No valid F1 scores to compute average.")
