# mobilenet模型训练代码，训练的模型会保存在models目录下，折线图会保存在results目录下

import tensorflow as tf
import matplotlib.pyplot as plt
from time import *


# 数据集加载函数，指明数据集的位置并统一处理为imgheight*imgwidth的大小，同时设置batch
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # 加载测试集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # 返回处理之后的训练集、验证集和类名
    return train_ds, val_ds, class_names


# 构建mobilenet模型
# 模型加载，指定图片处理的大小和是否进行迁移学习
def model_load(IMG_SHAPE=(224, 224, 3), class_num=3):
    # 微调的过程中不需要进行归一化的处理
    # 加载预训练的mobilenet模型
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    # 将模型的主干参数进行冻结
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        # 进行归一化的处理
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
        # 设置主干模型
        base_model,
        # 对主干模型的输出进行全局平均池化
        tf.keras.layers.GlobalAveragePooling2D(),
        # 通过全连接层映射到最后的分类数目上
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()
    # 模型训练的优化器为adam优化器，模型的损失函数为交叉熵损失函数
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 展示训练过程的曲线
def show_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy', history.history.get('val_acc', None))  # 兼容不同版本的键名变化
    loss = history.history['loss']
    val_loss = history.history.get('val_loss', None)

    if val_acc is not None and val_loss is not None:
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='test accuracy')
        plt.plot(val_acc, label='original accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('test and original accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='test loss')
        plt.plot(val_loss, label='original loss')
        plt.legend(loc='upper right')
        plt.ylabel('cross-entropy')
        plt.title('test and original loss')
        plt.xlabel('round')
        plt.savefig('results/results_mobilenet.png', dpi=100)
    else:
        print("No history records found for validation accuracy or validation loss.")


def train(epochs):
    # 开始训练，记录开始时间
    begin_time = time()
    # 加载数据集， 修改为你的数据集的路径
    train_ds, val_ds, class_names = data_load("/test",
                                              "/original", 224, 224, 16)
    print(class_names)
    # 加载模型
    model = model_load(class_num=len(class_names))
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # 保存模型， 修改为你要保存的模型的名称
    model.save("models/test_v_2.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")  # 该循环程序运行时间： 1.4201874732
    # 绘制模型训练过程图
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=30)
