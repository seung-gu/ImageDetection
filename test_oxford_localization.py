import os
import numpy as np
import tensorflow as tf
from processing.oxford_dataset_preprocessing import OxfordDataset
from processing.tf_record import TF_Record
from utils.display import predict_bbox
from utils.metrics import calculate_iou_bbox

cur_path = os.getcwd()

# oxford dataset load
oxford = OxfordDataset(dataset_dir='./dataset/oxford_pet')

dataset_dir, image_dir, bbox_dir, seg_dir = oxford.load_directory(os.path.join(cur_path, 'dataset'))
train_dir, val_dir, train_xml_files, val_xml_files = oxford.load_train_test_files(bbox_dir)

class_list = oxford.load_class_list(image_dir)
class2idx = {cls: idx for idx, cls in enumerate(class_list)}

tf_record = TF_Record(IMG_SIZE=224, N_TRAIN=len(train_xml_files), N_VAL=len(val_xml_files), N_BATCH=20)
mobile_net_model = tf.keras.models.load_model('./weights/mobile_net_v2_tl/local_output5.h5', compile=False)
out_features = [('bi_cls_num', tf.float32), ('x', tf.float32), ('y', tf.float32), ('w', tf.float32), ('h', tf.float32)]
# mobile_net_model = tf.keras.models.load_model('./weights/mobile_net_v2_tl/local_output4.h5', compile=False)
# out_features = [('x', tf.float32), ('y', tf.float32), ('w', tf.float32), ('h', tf.float32)]

tf_record.config_parse_function(in_features=tf_record.load_features(tfr_type='loc'), out_features=out_features)
tfr_train_dir, tfr_val_dir = tf_record.save_tf_record(dataset_dir, train_dir, val_dir, class2idx, tfr_type='loc')
train_dataset, val_dataset = tf_record.load_trainable_dataset(tfr_train_dir, tfr_val_dir, tf_record.parse_function)

steps_per_epoch = tf_record.N_TRAIN / tf_record.N_BATCH
validation_steps = int(np.ceil(tf_record.N_VAL / tf_record.N_BATCH))

# test
predict_bbox(mobile_net_model, tf_record, val_dataset=val_dataset, validation_steps=validation_steps)
calculate_iou_bbox(mobile_net_model, tf_record, val_dataset=val_dataset, validation_steps=validation_steps)
