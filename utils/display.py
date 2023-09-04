import numpy as np
import scipy as sp
import tensorflow as tf
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
app = QApplication(sys.argv)
window = QMainWindow()
window.show()


def show_cam(image_value, features, weights, results, label):
    '''
    Displays the class activation map of an image
    Args:
        image_value (tensor) -- preprocessed input image with size 224 x 224
        features (array) -- features of the image, shape (1, 7, 7, 512)
        results (array) -- output of the sigmoid layer
    '''

    features_for_img = features[0]
    prediction = results[0]
    class_activation_weigths = weights[:, label]
    class_activation_features = sp.ndimage.zoom(features_for_img, (224 / 7, 224 / 7, 1), order=2)
    cam_output = np.dot(class_activation_features, class_activation_weigths)
    cam_output = tf.reshape(cam_output, (224, 224))

    # visualize the results

    print(f'sigmoid output: {results}')
    print(f"prediction: {'dog' if tf.argmax(results[0]) else 'cat'}")

    plt.figure(figsize=(8, 8))
    plt.imshow(cam_output, cmap='jet', alpha=0.5)
    plt.imshow(tf.squeeze(image_value), alpha=0.5)
    plt.show()


def predict_bbox(model, tf_record, val_dataset, validation_steps):
    for val_data, val_gt in val_dataset.take(validation_steps):
        ## 정답 box 그리기
        x = val_gt[:, -4]
        y = val_gt[:, -3]
        w = val_gt[:, -2]
        h = val_gt[:, -1]
        xmin = x[0].numpy() - w[0].numpy() / 2.
        ymin = y[0].numpy() - h[0].numpy() / 2.
        rect_x = int(xmin * tf_record.IMG_SIZE)
        rect_y = int(ymin * tf_record.IMG_SIZE)
        rect_w = int(w[0].numpy() * tf_record.IMG_SIZE)
        rect_h = int(h[0].numpy() * tf_record.IMG_SIZE)

        rect = Rectangle((rect_x, rect_y), rect_w, rect_h, fill=False, color='red')
        plt.axes().add_patch(rect)

        ## 예측 box 그리기
        ## validation set에 대해서 bounding box 예측
        prediction = model.predict(val_data)
        pred_x = prediction[:, -4]
        pred_y = prediction[:, -3]
        pred_w = prediction[:, -2]
        pred_h = prediction[:, -1]
        pred_xmin = pred_x[0] - pred_w[0] / 2.
        pred_ymin = pred_y[0] - pred_h[0] / 2.
        pred_rect_x = int(pred_xmin * tf_record.IMG_SIZE)
        pred_rect_y = int(pred_ymin * tf_record.IMG_SIZE)
        pred_rect_w = int(pred_w[0] * tf_record.IMG_SIZE)
        pred_rect_h = int(pred_h[0] * tf_record.IMG_SIZE)

        pred_rect = Rectangle((pred_rect_x, pred_rect_y), pred_rect_w, pred_rect_h,
                              fill=False, color='blue')
        plt.axes().add_patch(pred_rect)

        ## image와 bbox 함께 출력
        plt.imshow(val_data[0])
        plt.show()


def predict_seg(model, tf_record, val_dataset, test_round=1):
    ## num_imgs만큼 validation dataset에서 읽어서 정답과 예측값 확인
    for idx, (image, seg) in enumerate(val_dataset.take(test_round)):
        plt.figure(figsize=(17, 6 * test_round))
        plt.subplot(test_round, 3, idx * 3 + 1)
        plt.imshow(image[0])
        plt.subplot(test_round, 3, idx * 3 + 2)
        plt.imshow(seg[0, :, :, 0], vmin=0, vmax=1)

        plt.subplot(test_round, 3, idx * 3 + 3)
        ## validation data에 대한 예측값 생성
        prediction = model.predict(image)
        pred = np.zeros_like(prediction)
        ## 0.5이상은 1로 나머지는 0으로 변환
        thr = 0.5
        pred[prediction >= thr] = 1
        pred[prediction < thr] = 0
        plt.imshow(pred[0, :, :, 1])
        plt.show()
