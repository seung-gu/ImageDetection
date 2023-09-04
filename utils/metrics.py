import numpy as np


def calculate_iou_bbox(model, tf_record, val_dataset, validation_steps):
    avg_iou = 0
    num_imgs = validation_steps
    res = tf_record.N_VAL % tf_record.N_BATCH  # 5
    for i, (val_data, val_gt) in enumerate(val_dataset.take(num_imgs)):
        # correct bounding boxes
        flag = (i == validation_steps - 1)
        x = val_gt[:, -4]
        y = val_gt[:, -3]
        w = val_gt[:, -2]
        h = val_gt[:, -1]
        prediction = model.predict(val_data)
        pred_x = prediction[:, -4]
        pred_y = prediction[:, -3]
        pred_w = prediction[:, -2]
        pred_h = prediction[:, -1]
        for idx in range(tf_record.N_BATCH):
            if (flag):
                if idx == res:
                    flag = False
                    break
            xmin = int((x[idx].numpy() - w[idx].numpy() / 2.) * tf_record.IMG_SIZE)
            ymin = int((y[idx].numpy() - h[idx].numpy() / 2.) * tf_record.IMG_SIZE)
            xmax = int((x[idx].numpy() + w[idx].numpy() / 2.) * tf_record.IMG_SIZE)
            ymax = int((y[idx].numpy() + h[idx].numpy() / 2.) * tf_record.IMG_SIZE)

            pred_xmin = int((pred_x[idx] - pred_w[idx] / 2.) * tf_record.IMG_SIZE)
            pred_ymin = int((pred_y[idx] - pred_h[idx] / 2.) * tf_record.IMG_SIZE)
            pred_xmax = int((pred_x[idx] + pred_w[idx] / 2.) * tf_record.IMG_SIZE)
            pred_ymax = int((pred_y[idx] + pred_h[idx] / 2.) * tf_record.IMG_SIZE)

            if xmin > pred_xmax or xmax < pred_xmin:
                continue
            if ymin > pred_ymax or ymax < pred_ymin:
                continue
            w_union = np.max((xmax, pred_xmax)) - np.min((xmin, pred_xmin))
            h_union = np.max((ymax, pred_ymax)) - np.min((ymin, pred_ymin))
            w_inter = np.min((xmax, pred_xmax)) - np.max((xmin, pred_xmin))
            h_inter = np.min((ymax, pred_ymax)) - np.max((ymin, pred_ymin))

            w_sub1 = np.abs(xmax - pred_xmax)
            h_sub1 = np.abs(ymax - pred_ymax)
            w_sub2 = np.abs(xmin - pred_xmin)
            h_sub2 = np.abs(ymin - pred_ymin)

            iou = (w_inter * h_inter) / ((w_union * h_union) - (w_sub1 * h_sub1) - (w_sub2 * h_sub2))
            avg_iou += iou / tf_record.N_VAL

    print(avg_iou)


def calculate_iou_seg(model, tf_record, val_dataset, validation_steps):
    ## IOU 계산
    # 두 맵을 더했을때 1인 부분끼리 더해져서 2가 되는 부분은 교집합 부분
    # 1이상인 부분 (1, 2) 은 합집합 부분
    print("Calculating avg IOU...")
    avg_iou = 0
    for images, labels in val_dataset.take(validation_steps):
        preds = model.predict(images)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0

        psum = labels[..., 0] + preds[..., 1]

        union = np.array(psum)
        union[union > 1] = 1.
        union = np.sum(union, axis=1)
        union = np.sum(union, axis=1)

        inter = np.array(psum)
        inter[inter == 1] = 0.
        inter[inter > 1] = 1.
        inter = np.sum(inter, axis=1)
        inter = np.sum(inter, axis=1)

        iou = inter / union
        avg_iou += np.sum(iou) / tf_record.N_VAL

    print(avg_iou)
