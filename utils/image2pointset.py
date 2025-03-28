import torch
import cv2
import numpy as np
from config import cfg
import json
from datetime import datetime as dt
import utils.data_transforms
import os


'''#numpy version
def get_image_pointset(rendering_images, npoint=cfg.CONST.NPOINT):
    batch_size = imgs.shape[0]
    # search_value = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    all_points = np.empty(shape=(batch_size, npoint, 2), dtype=int)
    for index, img in enumerate(imgs):
        img_points = img >= search_value
        img_points = np.where(img_points[..., 0] & img_points[..., 1] & img_points[..., 2])
        img_points = np.concatenate([np.expand_dims(coord, axis=1) for coord in img_points], axis=1)
        sample_points = farthest_point_sample(torch.from_numpy(img_points), npoint)
        all_points[index] = np.expand_dims(img_points[sample_points], axis=0)
        sample_image = np.zeros(shape=(height, width), dtype=np.uint8)
        sample_image[image_points[sample_points][:, 0], image_points[sample_points][:, 1]] = 255
        cv2.imshow("img", sample_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return all_points'''


def get_image_pointset(image, npoint=cfg.CONST.NPOINT):
    """
    Input:
        image: images data, [height, width, channel]
        npoint: number of samples
    Return:
        all_points: sampled points, [npoint, 2]
    """
    # height, width, channel = image.shape
    # search_value = torch.tensor(data=[1.0, 1.0, 1.0], dtype=torch.float32)
    # all_points = torch.zeros(size=(batch_size, npoint, 2), dtype=torch.float32)
    image_points = torch.where(torch.mean(input=image, dim=2) >= 0.5)
    image_points = torch.stack(image_points, dim=1)
    sample_points = farthest_point_sample(image_points, npoint)
    return image_points[sample_points]


def farthest_point_sample(xy, npoint=cfg.CONST.NPOINT):
    """
    Input:
        xy: pointcloud data, [N, 2]
        npoint: number of samples
    Return:
        centroids: sampled point index, [npoint]
    """
    N, C = xy.shape
    centroids = torch.zeros(size=(npoint,), dtype=torch.long)  # .to(device)
    distance = (torch.ones(N) * 1e10).long()
    farthest = torch.randint(low=0, high=N, size=(1,), dtype=torch.long)
    if N > npoint:  # 2D图像的有效像素点数超过取样点数
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xy[farthest, :].view(1, 2)
            dist = torch.sum(input=(xy - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=-1)[1]
    else:  # 像素值不足取样
        centroids[0:N] = torch.randperm(N)  # 随机打乱原像素排序,保证取到所有有效值
        centroids[N:npoint] = torch.randint(low=0, high=N, size=(npoint - N,))  # 剩余取样点,随机取值
    return centroids


def dataset2point(json_file, output_dir):
    """
    Args:
        json_file: 存储图像文件路径的json文件
        output_dir: 输出图片点云坐标文件的路径
    Return:
        file_num: 转换图片数量
    """
    file_num = 0
    rendering_path = cfg.DATASETS.SHAPENET.RENDERING_PATH
    img_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Resize((cfg.CONST.IMG_H, cfg.CONST.IMG_W)),
        utils.data_transforms.Binarize(threshold=0.2),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    ])
    out_txt_path = output_dir + '/txt/%s/%s/00.txt'
    out_png_path = output_dir + '/png/%s/%s/00.png'
    # Load all taxonomies of the dataset
    with open(json_file, mode='r', encoding='utf-8') as file:
        dataset_taxonomy = json.loads(file.read())
        # Load data for each category
        for taxonomy in dataset_taxonomy:
            taxonomy_folder_name = taxonomy['taxonomy_id']
            print('[INFO] %s Collecting files of Taxonomy[Name=%s]' % (dt.now(), taxonomy['taxonomy_name']))
            samples = taxonomy['train'] + taxonomy['test'] + taxonomy['val']
            for sample_idx, sample_name in enumerate(samples):
                # Get file list of rendering images
                #img_file_path = rendering_path % (taxonomy_folder_name, sample_name, 00)
                img_file_path = '/home/file_Wdisk/csw/Pix2Vox_ori/Pix2Vox/datasets/image/02691156/8c3419a655600e5766cf1b4a8fc3914e/草图.png'
                if not os.path.exists(img_file_path):
                    print('[WARN] %s Ignore sample %s/%s since img files not exists.' % (dt.now(), taxonomy_folder_name, sample_name))
                    continue
                img = cv2.imread(img_file_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
                img_height, img_width, img_channel = img.shape
                img = img_transforms([img])[0]
                img_points = get_image_pointset(torch.from_numpy(img).float())
                out_txt_file = out_txt_path % (taxonomy_folder_name, sample_name)
                out_png_file = out_png_path % (taxonomy_folder_name, sample_name)
                if not os.path.exists(os.path.dirname(out_txt_file)):
                    os.makedirs(os.path.dirname(out_txt_file))
                if not os.path.exists(os.path.dirname(out_png_file)):
                    os.makedirs(os.path.dirname(out_png_file))
                with open(out_txt_file, mode='w', encoding='utf-8') as ofile:
                    for point_idx in range(img_points.shape[0]):
                        ofile.write('%d,%d\n' % (img_points[point_idx][0], img_points[point_idx][1]))
                img = np.ones((cfg.CONST.IMG_H, cfg.CONST.IMG_W, 3), dtype=np.uint8)
                img *= 255  # white background
                for point_idx in range(img_points.shape[0]):
                    img[img_points[point_idx][0], img_points[point_idx][1], 0:3] = 0
                cv2.imwrite(out_png_file, img)
                file_num += 1
    return file_num

def point2image(json_file, point_dir, output_dir):
    """
        将保存的采样点集数据转为图片
    """
    with open(json_file, mode='r', encoding='utf-8') as file:
        dataset_taxonomy = json.loads(file.read())
    txt_path = os.path.join(point_dir, '%s', '%s', '00.txt')
    out_path = os.path.join(output_dir, '%s', '%s', '00.png')
    # Load data for each category
    for taxonomy in dataset_taxonomy:
        taxonomy_folder_name = taxonomy['taxonomy_id']
        sample_names = taxonomy['test'] + taxonomy['train'] + taxonomy['val']
        print('[INFO] %s Collecting files of Taxonomy[Name=%s]' % (dt.now(), taxonomy['taxonomy_name']))
        for sample_name in sample_names:
            out_png_path = out_path % (taxonomy_folder_name, sample_name)
            in_txt_path = txt_path % (taxonomy_folder_name, sample_name)
            if not os.path.exists(os.path.dirname(out_png_path)):
                os.makedirs(os.path.dirname(out_png_path))
            with open(in_txt_path, mode='r', encoding='utf-8') as f:
                img_points = f.readlines()
            img = np.ones((cfg.CONST.IMG_H, cfg.CONST.IMG_W, 3), dtype=np.uint8)
            img *= 255  # white background
            #cv2.imwrite(out_png_path, img)
            #img = cv2.imread(out_png_path)
            for point in img_points:
                point = point.strip().split(',')
                #img[int(point[0]), int(point[1]), 0:3] = 0
                cv2.circle(img, (int(point[1]), int(point[0])), 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.imwrite(out_png_path, img)


def dataset2point(img_file_path, output_txt_path="txt"):
    """
    Args:
        img_file_path: 单个图像文件路径
        output_txt_path: 输出2D点云txt文件的路径
        cfg: 配置对象，包含图像尺寸等信息

    Return:
        bool: 是否成功转换图像
    """
    img_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Resize((cfg.CONST.IMG_H, cfg.CONST.IMG_W)),
        utils.data_transforms.Binarize(threshold=0.2),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    ])
    out_txt_path = output_txt_path + '/00.txt'

    if not os.path.exists(img_file_path):
        print('[WARN] %s File not found: %s' % (dt.now(), img_file_path))
        return False

    img = cv2.imread(img_file_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    img = img_transforms([img])[0]

    img_points = get_image_pointset(torch.from_numpy(img).float())

    fileName = output_txt_path
    if not os.path.exists(fileName):
        os.makedirs(output_txt_path)

    with open(out_txt_path, mode='w', encoding='utf-8') as ofile:
        for point_idx in range(img_points.shape[0]):
            ofile.write('%d,%d\n' % (img_points[point_idx][0], img_points[point_idx][1]))

    return True

if __name__ == '__main__':
    #dataset2point(json_file='../datasets/overview.json', output_dir='../datasets/image')
    #point2image(json_file='../datasets/overview.json', point_dir='../datasets/image', output_dir='../datasets/image')
    img = "F:\Datasets\ShapeNetRendering_contour\\02691156\\1a04e3eab45ca15dd86060f189eb133\\rendering\\10.png"
    dataset2point(img)