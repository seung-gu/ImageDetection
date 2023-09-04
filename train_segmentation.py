import os
import numpy as np
import tensorflow as tf

from models.unet import UNet
from processing.download_oxford_dataset import download_VGGNet_weights_Topless
from processing.oxford_dataset_preprocessing import OxfordDataset
from processing.tf_record import TF_Record
from utils.display import predict_seg
from utils.metrics import calculate_iou_seg

cur_path = os.getcwd()

# oxford dataset load
oxford = OxfordDataset(dataset_dir='./dataset/oxford_pet')

dataset_dir, image_dir, bbox_dir, seg_dir = oxford.load_directory(os.path.join(cur_path, 'dataset'))
train_dir, val_dir, train_seg_files, test_seg_files = oxford.load_train_test_files(seg_dir)

class_list = oxford.load_class_list(image_dir)
class2idx = {cls: idx for idx, cls in enumerate(class_list)}

tf_record = TF_Record(IMG_SIZE=224, N_TRAIN=len(train_seg_files), N_VAL=len(test_seg_files), N_BATCH=5)
tf_record.config_parse_function(in_features=tf_record.load_features(tfr_type='seg'),
                                out_features=[('seg', tf.float32)])
tfr_train_dir, tfr_val_dir = tf_record.save_tf_record(dataset_dir, train_dir, val_dir, class2idx, tfr_type='seg',
                                                      update=False)
train_dataset, val_dataset = tf_record.load_trainable_dataset(tfr_train_dir, tfr_val_dir, tf_record.parse_function)

steps_per_epoch = tf_record.N_TRAIN / tf_record.N_BATCH
validation_steps = int(np.ceil(tf_record.N_VAL / tf_record.N_BATCH))

# train
unet = UNet(IMG_SIZE=224, transfer_learning_model=download_VGGNet_weights_Topless())
unet.compile_model(unet.model, learning_rate=0.0001, steps_per_epoch=steps_per_epoch)
unet.train_model(unet.model, train_dataset, val_dataset, epochs=10,
                 steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
unet.model.save('./weights/unet_tl/seg_model_tl.h5')
# unet.model.load_weights('./weights/seg_model.h5')
predict_seg(unet.model, tf_record, val_dataset, test_round=1)
calculate_iou_seg(unet.model, tf_record, val_dataset, validation_steps=validation_steps)
