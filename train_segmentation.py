import os
import numpy as np
import tensorflow as tf
from processing.oxford_dataset_preprocessing import OxfordDataset
from models.mobile_net_v2 import LocalizationMobileNetV2, LocalizationTwoBranchMobileNetV2
from processing.tf_record import TF_Record
from utils.display import display_seg

cur_path = os.getcwd()

# oxford dataset load
oxford = OxfordDataset(dataset_dir='./dataset/oxford_pet')

dataset_dir, image_dir, bbox_dir, seg_dir = oxford.load_directory(os.path.join(cur_path, 'dataset'))
train_dir, val_dir, train_seg_files, test_seg_files = oxford.load_train_test_files(seg_dir)

class_list = oxford.load_class_list(image_dir)
class2idx = {cls: idx for idx, cls in enumerate(class_list)}

tf_record = TF_Record(IMG_SIZE=224, N_TRAIN=len(train_seg_files), N_VAL=len(test_seg_files), N_BATCH=20)
tf_record.config_parse_function(in_features=tf_record.load_features(tfr_type='seg'),
                                out_features=[('seg', tf.float32)])
tfr_train_dir, tfr_val_dir = tf_record.save_tf_record(dataset_dir, train_dir, val_dir, class2idx, tfr_type='seg')
train_dataset, val_dataset = tf_record.load_trainable_dataset(tfr_train_dir, tfr_val_dir, tf_record.parse_function)

display_seg(val_dataset, 3)
