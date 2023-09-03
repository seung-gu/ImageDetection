from abc import abstractmethod

import tensorflow as tf
from keras.layers import BatchNormalization, ReLU
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate


class MobileNetV2TopLess:
    def __init__(self, input_shape, output_class):
        self.mobile_net_v2_top_less = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        self.model = self.create_top_model(self.mobile_net_v2_top_less, output_class)

    @abstractmethod
    def create_top_model(self, mobile_net_v2_top_less, output_class):
        pass

    @abstractmethod
    def compile_model(self, model, learning_rate, steps_per_epoch):
        pass

    def train_model(self, model, train_dataset, val_dataset, epochs, steps_per_epoch, validation_steps):
        return model.fit(
            train_dataset, steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_dataset,
            validation_steps=validation_steps
        )


class TransferLearningMobileNetV2(MobileNetV2TopLess):
    def __init__(self, input_shape, output_class):
        super().__init__(input_shape, output_class)

    def create_top_model(self, mobile_net_v2_top_less, output_class):
        gap = GlobalAveragePooling2D()(mobile_net_v2_top_less.output)
        output = Dense(output_class, activation='softmax', name='output_layer')(gap)
        return keras.Model(inputs=mobile_net_v2_top_less.input, outputs=output)

    def compile_model(self, model, learning_rate, steps_per_epoch):
        # learning rate scheduling
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                                  decay_steps=steps_per_epoch * 5,
                                                                  decay_rate=0.5,
                                                                  staircase=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()


class LocalizationMobileNetV2(MobileNetV2TopLess):
    def __init__(self, input_shape, output_class=4):
        super().__init__(input_shape, output_class)

    def create_top_model(self, mobile_net_v2_top_less, output_class):
        model = keras.models.Sequential()
        model.add(mobile_net_v2_top_less)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dense(output_class, activation='sigmoid'))
        return model

    def loss_fn(self, y_true, y_pred):
        return keras.losses.MeanSquaredError()(y_true, y_pred)

    def compile_model(self, model, learning_rate, steps_per_epoch):
        ## learning rate scheduing
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                                  decay_steps=steps_per_epoch * 10,
                                                                  decay_rate=0.5,
                                                                  staircase=True)
        ## optimizer는 RMSprop, loss는 mean squared error 사용
        model.compile(optimizers.RMSprop(lr_schedule, momentum=0.9), loss=self.loss_fn)


class LocalizationTwoBranchMobileNetV2(MobileNetV2TopLess):
    def __init__(self, input_shape):
        super().__init__(input_shape, output_class=None)

    def create_top_model(self, mobile_net_v2_top_less, output_class):
        # functional model 이용 - Sequential model로는 순서대로 쌓을수 밖에 없어서 두개의 branch로 나뉘는 모델을 구현할 수 없음
        gap = GlobalAveragePooling2D()(self.mobile_net_v2_top_less.output)

        # branch 1 (for classification)
        dense_b1_1 = Dense(256)(gap)
        bn_b1_2 = BatchNormalization()(dense_b1_1)
        relu_b1_3 = ReLU()(bn_b1_2)
        dense_b1_4 = Dense(64)(relu_b1_3)
        bn_b1_5 = BatchNormalization()(dense_b1_4)
        relu_b1_6 = ReLU()(bn_b1_5)
        output1 = Dense(2, activation='softmax')(relu_b1_6)

        # branch 2 (앞의 localization network 그대로)
        dense_b2_1 = Dense(256)(gap)
        bn_b2_2 = BatchNormalization()(dense_b2_1)
        relu_b2_3 = ReLU()(bn_b2_2)
        dense_b2_4 = Dense(64)(relu_b2_3)
        bn_b2_5 = BatchNormalization()(dense_b2_4)
        relu_b2_6 = ReLU()(bn_b2_5)
        output2 = Dense(4, activation='sigmoid')(relu_b2_6)

        concat = Concatenate()([output1, output2])

        return keras.Model(inputs=self.mobile_net_v2_top_less.input, outputs=concat)

    def loss_fn(self, y_true, y_pred):
        cls_labels = tf.cast(y_true[:, :1], tf.int64)  # 0번째 col : class label
        loc_labels = y_true[:, 1:]
        cls_preds = y_pred[:, :2]  # 0, 1 번째 col : class predictions
        loc_preds = y_pred[:, 2:]
        cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()(cls_labels, cls_preds)
        loc_loss = tf.keras.losses.MeanSquaredError()(loc_labels, loc_preds)
        return cls_loss + 5 * loc_loss  # localization 을 더 잘 하기 위해 localization 에 가중치 부과

    def compile_model(self, model, learning_rate, steps_per_epoch):
        ## learning rate scheduing
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                                  decay_steps=steps_per_epoch * 10,
                                                                  decay_rate=0.5,
                                                                  staircase=True)
        ## optimizer는 RMSprop, loss는 mean squared error 사용
        model.compile(optimizers.RMSprop(lr_schedule, momentum=0.9), loss=self.loss_fn)
