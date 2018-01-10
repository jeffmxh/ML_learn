# -*- coding: utf-8 -*-

'''
利用MNIST数据集训练一个GAN,输入0-9数字以及一个随机数，输出一个0-9数字的图片

@author: Jeffmxh
'''

from __future__ import absolute_import
from __future__ import print_function

import pickle
import logging
from collections import defaultdict
from PIL import Image
import numpy as np

import keras
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import multiply, Input, merge
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def get_logger():
    ''' 设置日志文件格式

    Usage:
      logger = get_logger()
      logger.info('Some messages')
    '''
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    # 建立一个streamhandler来把日志打在CMD窗口上，级别为info以上
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 设置日志格式
    formatter = logging.Formatter('[%(levelname)-3s]%(asctime)s %(filename)s[line:%(lineno)d]:%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def build_generator(latent_size):
    '''Build a convnet as generator of gans

    The generator is used to generate an image
    with an image_class label and a random number
    through a convolutional network

    Args:
      latent_size

    Returns:
      A convnet of which the input is image_class
      and a random number and the output is an image
    '''
    cnn = Sequential()

    cnn.add(Dense(1024, input_shape=(latent_size,), activation='relu'))
    cnn.add(Dense(128 * 7 * 7, activation='relu'))
    cnn.add(Reshape((128, 7, 7)))

    # 上采样，图像尺寸变为14 x 14
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(256, (5, 5), padding='same',
                   activation='relu', kernel_initializer='glorot_normal'))

    # 上采样，图像尺寸变为28 x 28
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(128, (5, 5), padding='same',
                   activation='relu', kernel_initializer='glorot_normal'))

    # 规约到一个通道
    cnn.add(Conv2D(1, (2, 2), padding='same',
                   activation='tanh', kernel_initializer='glorot_normal'))

    # 生成模型的输入层，特征向量
    latent = Input(shape=(latent_size, ))

    # 生成模型的输入层，标记
    image_class = Input(shape=(1,), dtype='int32')

    cls = Flatten()(Embedding(10, latent_size, embeddings_initializer='glorot_normal')(image_class))

    h = multiply([latent, cls])

    fake_image = cnn(h)
    return Model(inputs=[latent, image_class], outputs=fake_image)


def build_discriminator():
    '''Build a convnet as discriminator of gans

    The discriminator is used to decide which class the image
    belongs to, and whether the image is a fake image which
    is geneterated by the generator defined before

    Returns:
      A convolutional neural network which can give out the
      true-false judgement and the classification result
    '''
    # 采用Leaky RELU 来替换标准的卷积神经网络中的激活函数
    cnn = Sequential()

    cnn.add(Conv2D(32, (3, 3), padding='same',
                   strides=(2, 2), input_shape=(1, 28, 28)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, 28, 28))
    features = cnn(image)

    # 有两个输出
    # 输出真假值：范围在0-1
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    # 辅助分类器，输出图片分类
    aux = Dense(10, activation='softmax', name='auxiliary')(features)
    return Model(inputs=image, outputs=[fake, aux])


if __name__ == '__main__':

    logger = get_logger()
    mnist = read_data_sets("/home/da/jupyter_notebook/tensorflow/MNIST_data/", one_hot=True)

    # 定义超参数
    NB_EPOCHS = 50
    BATCH_SIZE = 100
    LATENT_SIZE = 100

    # 优化器的学习率
    ADAM_LR = 0.0002
    ADAM_BETA_1 = 0.5

    # 构建判别网络
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=ADAM_LR, beta_1=ADAM_BETA_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # 构建生成网络
    generator = build_generator(LATENT_SIZE)
    generator.compile(
        optimizer=Adam(lr=ADAM_LR, beta_1=ADAM_BETA_1),
        loss='binary_crossentropy'
    )
    latent = Input(shape=[LATENT_SIZE, ])
    image_class = Input(shape=(1,), dtype='int32')

    # 生成虚假图片
    fake = generator([latent, image_class])

    # 生成组合模型
    discriminator.trainablea = False
    fake, aux = discriminator(fake)
    combined = Model(inputs=[latent, image_class], outputs=[fake, aux])

    combined.compile(
        optimizer=Adam(lr=ADAM_LR, beta_1=ADAM_BETA_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # 将mnist数据转化为(28, 28, 1)的维度
    X_train, y_train, X_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    y_train = np.array([np.argmax(x) for x in y_train])
    y_test = np.array([np.argmax(x) for x in y_test])

    X_train = (X_train - 0.5) * 2
    X_train = X_train.reshape(X_train.shape[0], 28, 28)
    X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test - 0.5) * 2
    X_test = X_test.reshape(X_test.shape[0], 28, 28)
    X_test = np.expand_dims(X_test, axis=1)
    nb_train, nb_test = X_train.shape[0], X_test.shape[0]


    train_history = defaultdict(list)
    test_history = defaultdict(list)


    for epoch in range(NB_EPOCHS):

        nb_batches = int(X_train.shape[0] / BATCH_SIZE)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            logger.info('Epoch {} of {}, step {} of {}'.format(epoch + 1, NB_EPOCHS, index + 1, nb_batches))

            # 产生一个批次的噪声数据
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, LATENT_SIZE))

            # 获取一个批次真实数据
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            label_batch = y_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]

            # 生成一些噪声标记
            sampled_labels = np.random.randint(0, 10, BATCH_SIZE)

            # 产生一个批次的虚假图片
            generated_images = generator.predict(
                [noise, sampled_labels.reshape(-1, 1)], verbose=0
            )

            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

            # 产生两个批次的噪声和标记
            noise = np.random.uniform(-1, 1, (2 * BATCH_SIZE, LATENT_SIZE))
            sampled_labels = np.random.randint(0, 10, 2 * BATCH_SIZE)

            # 训练生成模型来欺骗判别模型
            trick = np.ones(2 * BATCH_SIZE)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape(-1, 1)], [trick, sampled_labels]))
        logger.info('Testing for epoch {}'.format(epoch + 1))

        # 评估测试集

        # 产生一个新批次噪声
        noise = np.random.uniform(-1, 1, (nb_test, LATENT_SIZE))

        sampled_labels = np.random.randint(0, 10, nb_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape(-1, 1)], verbose=False
        )

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # 看判别模型是否能判别
        discriminator_test_loss = discriminator.evaluate(X, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # 创建两个批次新的噪声数据
        noise = np.random.uniform(-1, 1, (2 * nb_test, LATENT_SIZE))
        sampled_labels = np.random.randint(0, 10, 2 * nb_test)

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape(-1, 1)],
            [trick, sampled_labels], verbose=False
        )

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # 记录损失值等性能指标并输出
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format('component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)', *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)', *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)', *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)', *test_history['discriminator'][-1]))

        # 每个epoch保存一次权重
        generator.save_weights('params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights('params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # 生成一些可视化的虚假数字看演化过程
        noise = np.random.uniform(-1, 1, (100, LATENT_SIZE))

        sampled_labels = np.array([[i] * 10 for i in range(10)]).reshape(-1, 1)

        generated_images = generator.predict([noise, sampled_labels], verbose=0)

        # 整理到一个方格中
        img = (np.concatenate([r.reshape(-1, 28)
                              for r in np.split(generated_images, 10)],
                              axis=-1) * 127.5 + 127.5).astype(np.uint8)
        Image.fromarray(img).save('plot_epoch_{0:03d}_generated.png'.format(epoch))

    pickle.dump({'train':train_history, 'test':test_history}, open('acgan_history.pickle', 'wb'))
