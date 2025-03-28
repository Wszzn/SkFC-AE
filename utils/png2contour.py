import cv2.cv2 as cv2
import os
import json
from tqdm import tqdm
from datetime import datetime as dt

def readDataset():

    with open('../datasets/ShapeNet/ShapeNet.json', encoding='utf-8') as file:
        dataset_taxonomy = json.loads(file.read())
    files = []
    # Load data for each category
    for taxonomy in dataset_taxonomy:
        taxonomy_folder_name = taxonomy['taxonomy_id']
        samples = taxonomy['train']+taxonomy['test']+taxonomy['val']
        files.extend(get_files_of_taxonomy(taxonomy_folder_name, samples))
    return files


def get_files_of_taxonomy(taxonomy_folder_name, samples):
    RENDERING_PATH = '/home/file_Wdisk/dataset/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
    files_of_taxonomy = []
    with tqdm(samples, desc='read PNG path from %s ' %(taxonomy_folder_name)) as samples:
        for sample_name in samples:
            # Get file path of volumes
            img_file_path = RENDERING_PATH % (taxonomy_folder_name, sample_name, 0)
            img_folder = os.path.dirname(img_file_path)
            if not os.path.exists(img_folder):
                print('\n [WARN] %s Ignore sample %s/%s not exists.' %(dt.now(), taxonomy_folder_name, sample_name))
                continue

            total_views = len(os.listdir(img_folder))
            rendering_image_indexes = range(total_views)
            rendering_images_file_path = []
            for image_idx in rendering_image_indexes:
                img_file_path = RENDERING_PATH % (taxonomy_folder_name, sample_name, image_idx)
                if not os.path.exists(img_file_path):
                    continue

                rendering_images_file_path.append(img_file_path)

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'rendering_images': rendering_images_file_path,
            })

    return files_of_taxonomy


def getOutline():
    files = readDataset()
    OUTPUT_PATH = '/home/file_Wdisk/dataset/ShapeNet/ShapeNetRendering_contour/%s/%s/rendering'
    with tqdm(files, desc='png2contour') as files:
        for sample in files:
            output_folder = OUTPUT_PATH % (sample['taxonomy_name'], sample['sample_name'])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            for rendering_image in sample['rendering_images']:
                img = cv2.imread(rendering_image, cv2.IMREAD_UNCHANGED)
                contour = cv2.Canny(img, 10, 50)
                contour_png = cv2.cvtColor(src=contour, code=cv2.COLOR_GRAY2BGRA)
                # contour_png[:, :, 3] = contour # 只留下轮廓,其余像素透明
                contour_png[:, :, 0:3] = (255-contour_png[:, :, 0:3])  # 二值反转,变白底黑框
                output_file_path = os.path.join(output_folder, os.path.split(rendering_image)[-1])
                cv2.imwrite(filename=output_file_path, img=contour_png, params=[cv2.IMWRITE_PNG_COMPRESSION, 1])
                # png = cv2.imread(filename=output_file_path, flags=cv2.IMREAD_UNCHANGED)
                # cv2.imshow(winname='new_win', mat=png)
                # cv2.waitKey(0)


if __name__ == '__main__':
    getOutline()