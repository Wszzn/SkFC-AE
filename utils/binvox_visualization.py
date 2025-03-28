# -*- coding: utf-8 -*-
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D


def get_volume_views(volume, save_dir, filename, IoU, n_itr):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")
    ax.view_init(elev=5, azim=45)  # 旋转坐标系

    plt.axis('off')
    save_path = os.path.join(save_dir, '%04f-%s.png' % (IoU, filename))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return cv2.imread(save_path)
