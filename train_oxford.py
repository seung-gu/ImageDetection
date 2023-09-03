import os
import numpy as np
from processing.oxford_dataset_preprocessing import OxfordDataset
from models.mobile_net_v2 import TransferLearningMobileNetV2
from processing.tf_record import TF_Record

cur_path = os.getcwd()

# oxford dataset load
oxford = OxfordDataset()
dataset_dir, image_dir, bbox_dir, seg_dir = oxford.load_directory(os.path.join(cur_path, 'dataset'))
train_dir, val_dir, train_image_files, val_image_files = oxford.load_train_test_files(image_dir)
class_list = oxford.load_class_list(image_dir)
class2idx = {cls: idx for idx, cls in enumerate(class_list)}
tf_record = TF_Record(IMG_SIZE=224, N_TRAIN=len(train_image_files), N_VAL=len(val_image_files), N_BATCH=20, N_CLASS=37)
tfr_train_dir, tfr_val_dir = tf_record.save_tf_record(dataset_dir, train_dir, val_dir, class2idx, tfr_type='cls')
train_dataset, val_dataset = tf_record.load_trainable_dataset(tfr_train_dir, tfr_val_dir)

# train
steps_per_epoch = tf_record.N_TRAIN / tf_record.N_BATCH
validation_steps = int(np.ceil(tf_record.N_VAL / tf_record.N_BATCH))

mobile_net = TransferLearningMobileNetV2(input_shape=(tf_record.IMG_SIZE, tf_record.IMG_SIZE, 3),
                                         output_class=tf_record.N_CLASS)
mobile_net.compile_model(mobile_net.model, learning_rate=0.0001, steps_per_epoch=steps_per_epoch)
history = mobile_net.train_model(mobile_net.model, train_dataset, val_dataset, epochs=40,
                                 steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

print(history)
