import tensorflow as tf
import os

from processing.oxford_dataset_preprocessing import OxfordDataset
from processing.tf_record import TF_Record
from utils.display import show_cam


cur_path = os.getcwd()
# 37 classes model
# model = tf.keras.models.load_model('weights/mobile_net_v2_tl/class37.h5')
# 2 classes model
model = tf.keras.models.load_model('weights/mobile_net_v2_tl/class2.h5')

oxford = OxfordDataset(dataset_dir='./dataset/oxford_pet')
dataset_dir, image_dir, bbox_dir, seg_dir = oxford.load_directory(os.path.join(cur_path, 'dataset'))
train_dir, val_dir, train_image_files, val_image_files = oxford.load_train_test_files(image_dir)
class_list = oxford.load_class_list(image_dir)

# assign index to class
class2idx = {cls: idx for idx, cls in enumerate(class_list)}
print(class2idx)

tfr_type = 'cls'
tf_record = TF_Record(IMG_SIZE=224, N_TRAIN=len(train_image_files), N_VAL=len(val_image_files), N_BATCH=1, N_CLASS=2)
tfr_train_dir, tfr_val_dir = tf_record.save_tf_record(dataset_dir, train_dir, val_dir, class2idx, tfr_type=tfr_type)
tf_record.config_parse_function(tf_record.load_features(tfr_type), ('cls_num', tf.int64))  # 37 classes
#tf_record.config_parse_function(tf_record.load_features(tfr_type), [('bi_cls_num', tf.int64)])  # 2 classes
train_dataset, val_dataset = tf_record.load_trainable_dataset(tfr_train_dir, tfr_val_dir, tf_record.parse_function)

# cam model
cam_model = tf.keras.Model(model.input, outputs=(model.layers[-3].output, model.layers[-1].output))
cam_model.summary()
gap_weights = model.layers[-1].get_weights()[0]
print(gap_weights.shape)

for img, lbl in val_dataset.take(5):
    print(f"ground truth: {'dog' if lbl else 'cat'}")
    features, results = cam_model.predict(img)
    show_cam(img, features, gap_weights, results, lbl)
