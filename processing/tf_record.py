import os
import re
import random

import tensorflow as tf
from PIL import Image
import xml.etree.ElementTree as et


# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class TF_Record:
    def __init__(self, IMG_SIZE, N_TRAIN, N_VAL, N_BATCH, N_CLASS=None):
        self.IMG_SIZE = IMG_SIZE
        self.N_TRAIN = N_TRAIN
        self.N_VAL = N_VAL
        self.N_BATCH = N_BATCH
        self.N_CLASS = N_CLASS

    def load_trainable_dataset(self, tfr_train_dir, tfr_val_dir, parse_func):
        # train dataset 만들기
        train_dataset = tf.data.TFRecordDataset(tfr_train_dir)
        train_dataset = train_dataset.map(parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.N_TRAIN). \
            prefetch(tf.data.experimental.AUTOTUNE).batch(self.N_BATCH).repeat()

        # val dataset 만들기
        val_dataset = tf.data.TFRecordDataset(tfr_val_dir)
        val_dataset = val_dataset.map(parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(self.N_BATCH).repeat()

        return train_dataset, val_dataset

    def config_parse_function(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

    def parse_function(self, tfrecord_serialized):
        if self.in_features is None or self.out_features is None:
            raise ValueError('You must set config for parse function')
        parsed_features = tf.io.parse_single_example(tfrecord_serialized, self.in_features)

        image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.reshape(image, [self.IMG_SIZE, self.IMG_SIZE, 3])
        image = tf.cast(image, tf.float32) / 255.

        output = tf.stack([tf.cast(parsed_features[cls], typ) for (cls, typ) in self.out_features], -1)
        print(output.shape)
        if output.shape[-1] == 1:
            output = tf.squeeze(output, -1)
        return image, output

    def save_tf_record(self, data_dir, train_dir, val_dir, class2idx, tfr_type='cls'):
        # tfr_type : 'cls' or 'loc'

        # create tfrecord directory if not exist
        tfr_dir = os.path.join(data_dir, 'tfrecord')
        os.makedirs(tfr_dir, exist_ok=True)

        tfr_train_path = os.path.join(tfr_dir, tfr_type + '_train.tfr')
        tfr_val_path = os.path.join(tfr_dir, tfr_type + '_val.tfr')

        # create tfrecord file if not exist
        if not os.path.exists(tfr_train_path) or not os.path.getsize(tfr_train_path):  # if file is empty
            writer_train = tf.io.TFRecordWriter(tfr_train_path)
            getattr(self, 'write_tf_record_' + tfr_type)(writer_train, data_dir, train_dir, class2idx)
        if not os.path.exists(tfr_val_path) or not os.path.getsize(tfr_val_path):
            writer_val = tf.io.TFRecordWriter(tfr_val_path)
            getattr(self, 'write_tf_record_' + tfr_type)(writer_val, data_dir, val_dir, class2idx)

        return tfr_train_path, tfr_val_path

    def write_tf_record_cls(self, writer, data_dir, image_dir, class2idx):
        files = os.listdir(image_dir)
        print("Writing TF Record " + str(len(files)) + " images from " + image_dir)
        for file in files:
            path = os.path.join(image_dir, file)
            image = Image.open(path)
            image = image.resize((self.IMG_SIZE, self.IMG_SIZE))
            bimage = image.tobytes()

            file_name = os.path.splitext(file)[0]  # Bangal_101
            class_name = re.sub('_\d+', '', file_name)
            class_num = class2idx[class_name]

            if file_name[0].islower():  # dog
                bi_cls_num = 0
            else:  # cat
                bi_cls_num = 1

            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(bimage),
                'cls_num': _int64_feature(class_num),
                'bi_cls_num': _int64_feature(bi_cls_num)
            }))
            writer.write(example.SerializeToString())

        writer.close()

    def write_tf_record_loc(self, writer, data_dir, bbox_dir, class2idx):
        files = os.listdir(bbox_dir)
        print("Writing TF Record " + str(len(files)) + " xmls from " + bbox_dir)
        for bbox_file in files:
            bbox_path = os.path.join(bbox_dir, bbox_file)

            tree = et.parse(bbox_path)
            width = float(tree.find('./size/width').text)
            height = float(tree.find('.size/height').text)
            xmin = float(tree.find('./object/bndbox/xmin').text)
            ymin = float(tree.find('./object/bndbox/ymin').text)
            xmax = float(tree.find('./object/bndbox/xmax').text)
            ymax = float(tree.find('./object/bndbox/ymax').text)
            xc = (xmin + xmax) / 2.
            yc = (ymin + ymax) / 2.
            x = xc / width
            y = yc / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            file_name = os.path.splitext(bbox_file)[0]
            image_file = file_name + '.jpg'
            image_dir = os.path.join(data_dir, 'images')
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path)
            image = image.resize((self.IMG_SIZE, self.IMG_SIZE))
            bimage = image.tobytes()

            class_name = re.sub('_\d+', '', file_name)
            class_num = class2idx[class_name]

            if file_name[0].islower():  # dog
                bi_cls_num = 0
            else:  # cat
                bi_cls_num = 1

            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(bimage),
                'cls_num': _int64_feature(class_num),
                'bi_cls_num': _int64_feature(bi_cls_num),
                'x': _float_feature(x),
                'y': _float_feature(y),
                'w': _float_feature(w),
                'h': _float_feature(h)
            }))
            writer.write(example.SerializeToString())

        writer.close()

    def load_features(self, tfr_type='cls'):
        if tfr_type == 'cls':
            return {'image': tf.io.FixedLenFeature([], tf.string),
                    'cls_num': tf.io.FixedLenFeature([], tf.int64),
                    'bi_cls_num': tf.io.FixedLenFeature([], tf.int64)
                    }
        elif tfr_type == 'loc':
            return {'image': tf.io.FixedLenFeature([], tf.string),
                    'cls_num': tf.io.FixedLenFeature([], tf.int64),
                    'bi_cls_num': tf.io.FixedLenFeature([], tf.int64),
                    'x': tf.io.FixedLenFeature([], tf.float32),
                    'y': tf.io.FixedLenFeature([], tf.float32),
                    'w': tf.io.FixedLenFeature([], tf.float32),
                    'h': tf.io.FixedLenFeature([], tf.float32)
                    }
