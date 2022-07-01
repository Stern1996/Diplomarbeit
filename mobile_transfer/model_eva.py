import datetime

from keras.models import *
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder

#先用目标域少量样本训练后再预测
def transfer_mobile_pre():
    # 每个样本为（64，64）
    datasets, labels = np.load('./data/MFPT_database.npy'), np.load('./data/MFPT_labels.npy')
    # #标签集one-hot编码,此时labels为（样本数x4）
    enc = OneHotEncoder()
    enc.fit([[0], [1], [2], [3]])
    labels = enc.transform(labels).toarray()
    datasets = tf.keras.applications.mobilenet.preprocess_input(
            datasets, data_format=None
        )
    labels = labels.reshape((labels.shape[0], 1, 1, -1))
    # 训练集占比0.2，测试集占比0.8
    X_train, X_test, Y_train, Y_test = train_test_split(datasets, labels, test_size=0.8, random_state=0)

    # 设置tensorboard相关参数
    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=(2,50))

    #加载模型
    model = load_model('../MobileNet/LeNet5/Mobile_Lenet5.h5')

    #冻结最后两层
    layers= model.layers
    for layer in layers:
        layer.trainable = False
    layers[-1].trainable = True
    layers[-2].trainable = True
    # layers[-3].trainable = True
    # layers[-4].trainable = True
    #倒数第4层
    # layers[-5].trainable = True
    # layers[-6].trainable = True
    # layers[-7].trainable = True
    # layers[-8].trainable = True
    # layers[-9].trainable = True
    # layers[-10].trainable = True
    #倒数第5层
    # layers[-11].trainable = True
    # layers[-12].trainable = True
    # layers[-13].trainable = True
    # layers[-14].trainable = True
    # layers[-15].trainable = True
    # layers[-16].trainable = True
    # 倒数第6层
    layers[-17].trainable = True
    layers[-18].trainable = True
    layers[-19].trainable = True
    layers[-20].trainable = True
    layers[-21].trainable = True
    layers[-22].trainable = True
    # 倒数第7层
    # layers[-23].trainable = True
    # layers[-24].trainable = True
    # layers[-25].trainable = True
    # layers[-26].trainable = True
    # layers[-27].trainable = True
    # layers[-28].trainable = True
    # 倒数第8层
    # layers[-29].trainable = True
    # layers[-30].trainable = True
    # layers[-31].trainable = True

    #迁移预训练
    model.summary()
    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    #分别再用504个样本中不同数量样本用于一边训练一边验证
    history = model.fit(X_train, Y_train, epochs=120, batch_size=16, validation_split=0.5,callbacks=tensorboard_callback)
    model.evaluate(X_test, Y_test)
    model.save("./pre_mobile_Lenet5.h5")

    # 训练过程可视化
    fig = plt.figure()  # 新建一张图
    plt.plot(history.history['accuracy'], label='training acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    fig.savefig('pre_Mobile_pre_' + 'acc.png')
    fig = plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig('pre_Mobile_pre_' + 'loss.png')

#加载模型直接评估
def model_direct():
    # 加载模型
    model = load_model('./small_mobile_Lenet5.h5')
    # MFPT全数据集
    datasets2, labels2 = np.load('./data/small_MFPT_database.npy'), np.load('./data/small_MFPT_labels.npy')
    # #标签集one-hot编码,此时labels为（样本数x4）
    enc = OneHotEncoder()
    enc.fit([[0], [1], [2], [3]])
    labels2 = enc.transform(labels2).toarray()
    datasets2 = tf.keras.applications.mobilenet.preprocess_input(
        datasets2, data_format=None
    )
    labels2 = labels2.reshape((labels2.shape[0], 1, 1, -1))
    #评估模型
    model.evaluate(datasets2, labels2)



if __name__ == "__main__":
    transfer_mobile_pre()
    #model_direct()