from time import time

import matplotlib.pyplot as plt
import tensorflow as tf


# 数据集加载函数，指明数据集的位置并统一处理为img_height*img_width的大小，同时设置batch
def data_load(train_dir, val_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


# 模型加载，指定图片处理的大小和是否进行迁移学习
def model_load(IMG_SHAPE=(224, 224, 3), class_num=5):
    # 创建 ResNet50 主干模型
    base_model_resnet = tf.keras.applications.ResNet50(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model_resnet.trainable = False  # 冻结 ResNet50 的权重

    # 创建 MobileNetV2 主干模型
    base_model_mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model_mobilenet.trainable = False  # 冻结 MobileNetV2 的权重

    # 定义模型的输入
    input_layer = tf.keras.layers.Input(shape=IMG_SHAPE)

    # 对输入进行归一化处理
    normalized_input = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)(input_layer)

    # 分别使用 ResNet50 和 MobileNetV2 处理输入
    resnet_output = base_model_resnet(normalized_input)
    mobilenet_output = base_model_mobilenet(normalized_input)

    # 对每个主干模型的输出进行全局平均池化
    resnet_output = tf.keras.layers.GlobalAveragePooling2D()(resnet_output)
    mobilenet_output = tf.keras.layers.GlobalAveragePooling2D()(mobilenet_output)

    # 使用 Concatenate 层连接两个主干模型的输出
    concatenated_output = tf.keras.layers.Concatenate()([resnet_output, mobilenet_output])

    # 通过全连接层映射到最后的分类数目上
    output_layer = tf.keras.layers.Dense(class_num, activation='softmax')(concatenated_output)

    # 构建模型
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # 打印模型结构
    model.summary()

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 展示训练过程的曲线
def show_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy', history.history.get('val_acc', None))
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
        # 保存结果图
        plt.savefig('../results/res_mobilenet.png', dpi=100)
    else:
        print("No history records found for validation accuracy or validation loss.")


# 训练函数
def train(epochs):
    # 开始训练，记录开始时间
    begin_time = time()
    # 加载数据集， 修改为你的数据集的路径
    train_ds, val_ds, class_names = data_load("../datasets/test",
                                              "../datasets/val",
                                              224, 224, 16)
    print(class_names)
    # 加载模型
    model = model_load(class_num=len(class_names))
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # 保存模型， 修改为你要保存的模型的名称
    model.save("../models/res_mobile.keras")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")
    # 绘制模型训练过程图
    show_loss_acc(history)
    return val_ds


if __name__ == '__main__':
    val = train(epochs=20)

    labels = ['broke', 'circle', 'good', 'lose', 'uncovered']
