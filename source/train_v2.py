import tensorflow as tf
import matplotlib.pyplot as plt
from time import time


# 数据集加载函数，使用生成器逐批次加载数据
def data_generator(data_dir, img_height, img_width, batch_size):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 127.5, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')
    return train_generator, val_generator, train_generator.class_indices


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
def train(epochs, train_generator, val_generator):
    # 开始训练，记录开始时间
    begin_time = time()
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)
    # 保存模型， 修改为你要保存的模型的名称
    model.save("../models/res_mobile.keras")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")
    # 绘制模型训练过程图
    show_loss_acc(history)


if __name__ == '__main__':
    train_dir = "../datasets/train"
    img_height = 224
    img_width = 224
    batch_size = 16
    epochs = 1

    train_generator, val_generator, class_indices = data_generator(train_dir, img_height, img_width, batch_size)
    model = model_load(class_num=len(class_indices))
    train(epochs, train_generator, val_generator)
