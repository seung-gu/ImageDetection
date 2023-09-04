from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, MaxPooling2D, Dense, BatchNormalization, \
    GlobalAveragePooling2D, Concatenate
from tensorflow.keras.utils import get_file

from processing.download_oxford_dataset import download_VGGNet_weights_Topless


class UNet:
    def __init__(self, IMG_SIZE=224, transfer_learning_model=None):
        self.IMG_SIZE = IMG_SIZE
        self.model = self.create_model(transfer_learning_model)

    def create_model(self, transfer_learning_model):
        # encoder
        inputs = Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))

        conv1_1 = Conv2D(64, 3, 1, 'SAME', activation='relu')(inputs)
        conv1_2 = Conv2D(64, 3, 1, 'SAME', activation='relu')(conv1_1)
        pool1_3 = MaxPooling2D()(conv1_2)

        conv2_1 = Conv2D(128, 3, 1, 'SAME', activation='relu')(pool1_3)
        conv2_2 = Conv2D(128, 3, 1, 'SAME', activation='relu')(conv2_1)
        pool2_3 = MaxPooling2D()(conv2_2)

        conv3_1 = Conv2D(256, 3, 1, 'SAME', activation='relu')(pool2_3)
        conv3_2 = Conv2D(256, 3, 1, 'SAME', activation='relu')(conv3_1)
        conv3_3 = Conv2D(256, 3, 1, 'SAME', activation='relu')(conv3_2)
        pool3_4 = MaxPooling2D()(conv3_3)

        conv4_1 = Conv2D(512, 3, 1, 'SAME', activation='relu')(pool3_4)
        conv4_2 = Conv2D(512, 3, 1, 'SAME', activation='relu')(conv4_1)
        conv4_3 = Conv2D(512, 3, 1, 'SAME', activation='relu')(conv4_2)
        pool4_4 = MaxPooling2D()(conv4_3)

        conv5_1 = Conv2D(512, 3, 1, 'SAME', activation='relu')(pool4_4)
        conv5_2 = Conv2D(512, 3, 1, 'SAME', activation='relu')(conv5_1)
        conv5_3 = Conv2D(512, 3, 1, 'SAME', activation='relu')(conv5_2)
        pool5_4 = MaxPooling2D()(conv5_3)

        # loading vgg16 pretrained weights
        if transfer_learning_model:
            print("Loading pretrained weights from", transfer_learning_model)
            # 인코더 부분만 pretrained 모델 사용, 디코더 부분은 그냥 random initialization 이용
            # segmentation에선 인코더 부분이 훨씬 중요하기 때문에 디코더 부분까지도 pretrained 모델
            # 안써도 되고 이렇게 해도 됨
            vgg = keras.Model(inputs, pool5_4)  # model inputs to pool5_4
            vgg.load_weights(transfer_learning_model)

        # decoder
        upconv6 = Conv2DTranspose(512, 5, 2, 'SAME', activation='relu')(pool5_4)
        concat6 = Concatenate()([conv5_3, upconv6])
        conv6 = Conv2D(512, 3, 1, 'SAME', activation='relu')(concat6)

        upconv7 = Conv2DTranspose(512, 5, 2, 'SAME', activation='relu')(conv6)
        concat7 = Concatenate()([conv4_3, upconv7])
        conv7 = Conv2D(512, 3, 1, 'SAME', activation='relu')(concat7)

        upconv8 = Conv2DTranspose(256, 5, 2, 'SAME', activation='relu')(conv7)
        concat8 = Concatenate()([conv3_3, upconv8])
        conv8 = Conv2D(256, 3, 1, 'SAME', activation='relu')(concat8)

        upconv9 = Conv2DTranspose(128, 5, 2, 'SAME', activation='relu')(conv8)
        concat9 = Concatenate()([conv2_2, upconv9])
        conv9 = Conv2D(128, 3, 1, 'SAME', activation='relu')(concat9)

        upconv10 = Conv2DTranspose(64, 5, 2, 'SAME', activation='relu')(conv9)
        concat10 = Concatenate()([conv1_2, upconv10])
        conv10 = Conv2D(64, 3, 1, 'SAME', activation='relu')(concat10)

        conv11 = Conv2D(64, 3, 1, 'SAME', activation='relu')(conv10)
        # 64 -> 2 채널로 각 Seg 맵이 고양이나 강아지의 pixel 을 보여줌
        conv12 = Conv2D(2, 1, 1, 'SAME', activation='softmax')(conv11)

        return keras.Model(inputs=inputs, outputs=conv12)

    def compile_model(self, model, learning_rate, steps_per_epoch):
        ## learning rate scheduing
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                                  decay_steps=steps_per_epoch * 10,
                                                                  decay_rate=0.4,
                                                                  staircase=True)
        ## optimizer는 Adam, loss는 sparse categorical crossentropy 사용
        ## label이 ont-hot으로 encoding 안 된 경우에 sparse categorical corssentropy 및 sparse categorical accuracy 사용
        model.compile(keras.optimizers.Adam(lr_schedule), loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])
        model.summary()

    def train_model(self, model, train_dataset, val_dataset, epochs, steps_per_epoch, validation_steps):
        return model.fit(
            train_dataset, steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_dataset,
            validation_steps=validation_steps
        )
