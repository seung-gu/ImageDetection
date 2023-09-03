import os
import re
import shutil

import numpy as np
from PIL import Image


def get_class_distribution(files):
    # count average number of files of each class
    class_dist = dict()  # class distribution
    for file in files:
        file_name = os.path.splitext(file)[0]
        class_name = re.sub('_\d+', '', file_name)
        if class_name in class_dist:
            class_dist[class_name] += 1
        else:
            class_dist[class_name] = 1
    return class_dist


class OxfordDataset:
    def __init__(self, dataset_dir='./dataset/oxford_pet'):
        if not os.path.exists(dataset_dir):
            raise ValueError('Dataset path not exists.')

        self.dataset_dir = dataset_dir

        if not os.path.exists(os.path.join(dataset_dir, 'train')):
            image_dir = os.path.join(dataset_dir, 'images')
            bbox_dir = os.path.join(dataset_dir, 'annotations', 'xmls')
            seg_dir = os.path.join(dataset_dir, 'annotations', 'trimaps')
            self.synchronize_image_with_annotations(image_dir, bbox_dir, seg_dir)

    def synchronize_image_with_annotations(self, image_dir, bbox_dir, seg_dir):
        # first time when dataset is downloaded, delete images except 3 RGB channel images
        image_files = [fname for fname in os.listdir(image_dir) if os.path.splitext(fname)[-1] == '.jpg']
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            bbox_file = os.path.splitext(image_file)[0] + '.xml'
            bbox_path = os.path.join(bbox_dir, bbox_file)
            seg_file = os.path.splitext(image_file)[0] + '.png'
            seg_path = os.path.join(seg_dir, seg_file)
            image = Image.open(image_path)
            image_mode = image.mode
            if image_mode != 'RGB':
                print("deleted :", image_file, image_mode)
                image = np.asarray(image)
                print(image.shape)
                try:
                    os.remove(image_path)
                    os.remove(bbox_path)
                    os.remove(seg_path)
                except FileNotFoundError:
                    pass

        bbox_files = [fname for fname in os.listdir(bbox_dir) if os.path.splitext(fname)[-1] == '.xml']
        seg_files = [fname for fname in os.listdir(seg_dir) if os.path.splitext(fname)[-1] == '.png']

        for bbox_file in bbox_files:
            image_file = os.path.splitext(bbox_file)[0] + '.jpg'
            if not os.path.exists(os.path.join(image_dir, image_file)):
                print("deleted :", bbox_file)
                os.remove(os.path.join(bbox_dir, bbox_file))

        for seg_file in seg_files:
            image_file = os.path.splitext(seg_file)[0] + '.jpg'
            if not os.path.exists(os.path.join(image_dir, image_file)):
                print("deleted :", seg_file)
                os.remove(os.path.join(seg_dir, seg_file))

    def load_directory(self, path):
        dataset_dir = os.path.join(path, 'oxford_pet')
        image_dir = os.path.join(dataset_dir, 'images')
        bbox_dir = os.path.join(dataset_dir, 'annotations', 'xmls')
        seg_dir = os.path.join(dataset_dir, 'annotations', 'trimaps')

        return dataset_dir, image_dir, bbox_dir, seg_dir

    def load_train_test_files(self, dir):
        if dir.endswith('images'):
            train_dir = os.path.join(self.dataset_dir, 'train')
            val_dir = os.path.join(self.dataset_dir, 'val')
        elif dir.endswith('xmls'):
            train_dir = os.path.join(self.dataset_dir, 'train_xml')
            val_dir = os.path.join(self.dataset_dir, 'val_xml')
        elif dir.endswith('trimaps'):
            train_dir = os.path.join(self.dataset_dir, 'train_seg')
            val_dir = os.path.join(self.dataset_dir, 'val_seg')
        else:
            raise ValueError('Invalid directory path.')

        if os.path.exists(train_dir) and os.listdir(train_dir):
            pass
        else:
            self.split_train_test_files(dir, train_dir, val_dir)

        return train_dir, val_dir, os.listdir(train_dir), os.listdir(val_dir)

    def load_class_list(self, image_dir):
        image_files = [fname for fname in os.listdir(image_dir) if os.path.splitext(fname)[-1] == '.jpg']
        class_list = set()
        for image_file in image_files:
            file_name = os.path.splitext(image_file)[0]
            class_name = re.sub('_\d+', '', file_name)
            class_list.add(class_name)
        class_list = list(class_list)
        class_list.sort()
        print("Number of classes : ", len(class_list))
        print(class_list)
        return class_list

    # split image files and copy to train or val directory
    def split_train_test_files(self, dir, train_dir, val_dir):
        # create train, val directory for the first time
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        if dir.endswith('xmls'):
            files = [fname for fname in os.listdir(dir) if os.path.splitext(fname)[-1] == '.xml']
        elif dir.endswith('trimaps'):
            files = [fname for fname in os.listdir(dir) if os.path.splitext(fname)[-1] == '.png']
        else:
            files = [fname for fname in os.listdir(dir) if os.path.splitext(fname)[-1] == '.jpg']
        files.sort()
        print("Number of files : ", len(files))

        cnt = 0
        train_cnt = round(np.mean(list(get_class_distribution(files).values())) * 0.8)
        previous_class = ""
        for file in files:
            file_name = os.path.splitext(file)[0]
            class_name = re.sub('_\d+', '', file_name)
            if class_name == previous_class:
                cnt += 1
            else:
                cnt = 1
            if cnt <= train_cnt:
                cpath = train_dir
            else:
                cpath = val_dir
            path = os.path.join(dir, file)
            shutil.copy(path, cpath)  # copy image file to train or val directory
            previous_class = class_name
