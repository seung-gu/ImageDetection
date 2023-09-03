import os
import numpy as np
import tensorflow as tf
from processing.oxford_dataset_preprocessing import OxfordDataset
from models.mobile_net_v2 import LocalizationMobileNetV2, LocalizationTwoBranchMobileNetV2
from processing.tf_record import TF_Record

cur_path = os.getcwd()

# oxford dataset load
oxford = OxfordDataset(dataset_dir='./dataset/oxford_pet')

dataset_dir, image_dir, bbox_dir, seg_dir = oxford.load_directory(os.path.join(cur_path, 'dataset'))
train_dir, val_dir, train_xml_files, val_xml_files = oxford.load_train_test_files(bbox_dir)

class_list = oxford.load_class_list(image_dir)
class2idx = {cls: idx for idx, cls in enumerate(class_list)}

tf_record = TF_Record(IMG_SIZE=224, N_TRAIN=len(train_xml_files), N_VAL=len(val_xml_files), N_BATCH=20)
tf_record.config_parse_function(in_features=tf_record.load_features(tfr_type='loc'),
                                out_features=[('bi_cls_num', tf.float32), ('x', tf.float32), ('y', tf.float32),
                                              ('w', tf.float32), ('h', tf.float32)])
# out_features=[('x', tf.float32), ('y', tf.float32), ('w', tf.float32), ('h', tf.float32)])
tfr_train_dir, tfr_val_dir = tf_record.save_tf_record(dataset_dir, train_dir, val_dir, class2idx, tfr_type='loc')
train_dataset, val_dataset = tf_record.load_trainable_dataset(tfr_train_dir, tfr_val_dir, tf_record.parse_function)

steps_per_epoch = tf_record.N_TRAIN / tf_record.N_BATCH
validation_steps = int(np.ceil(tf_record.N_VAL / tf_record.N_BATCH))

# train
mobile_net = LocalizationTwoBranchMobileNetV2(input_shape=(tf_record.IMG_SIZE, tf_record.IMG_SIZE, 3))
# mobile_net = LocalizationMobileNetV2(input_shape=(tf_record.IMG_SIZE, tf_record.IMG_SIZE, 3))
mobile_net.compile_model(mobile_net.model, learning_rate=0.0001, steps_per_epoch=steps_per_epoch)
history = mobile_net.train_model(mobile_net.model, train_dataset, val_dataset, epochs=40,
                                 steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
print(history)
