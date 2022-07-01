#import all necessary layers
import datetime
import tensorflow as tf
from keras.layers import Input, DepthwiseConv2D
from keras.layers import Conv2D, BatchNormalization
from keras.layers import ReLU, AvgPool2D, Flatten, Dense,Dropout
from keras import Model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 每个样本为（64，64），一共4800个
datasets, labels = np.load('./data/database.npy'), np.load('./data/labels.npy')
#标签集one-hot编码,此时labels为（样本数x4）
enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()
datasets = tf.keras.applications.mobilenet.preprocess_input(
    datasets, data_format=None
)
labels = labels.reshape((labels.shape[0],1,1,-1))
# 数据集划分，训练集和验证集个3840，测试集960个
X_train, X_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2, random_state=0)

# MobileNet block
def mobilnet_block(x, filters, strides):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

#stem of the model
input = Input(shape = (64,64,1))
x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input)
x = BatchNormalization()(x)
x = ReLU()(x)

#LeNet5结构： Input(32x32x1) -> Conv2(5x5x32) -> Pooling(2x2) -> Conv2(5x5x16) -> Pooling(2x2) -> Conv2(5x5x120) -> FC(84) -> FC(10)
#故类似MobileNet的LeNet5改版： Conv/s2 -- block1 -- block2 -- block3 -- block4 -- Pool -- FC(84) -- FC(4)
# main part of the model(slim版)
x = mobilnet_block(x, filters = 48, strides = 1)
x = mobilnet_block(x, filters = 96, strides = 2)
x = mobilnet_block(x, filters = 96, strides = 1)
x = mobilnet_block(x, filters = 192, strides = 2)

x = AvgPool2D (pool_size = 8, strides = 1)(x)
x = Dense (units = 84, activation = 'relu')(x)
x = Dropout(0.3)(x)
output = Dense (units = 4, activation = 'softmax')(x)

model = Model(inputs=input, outputs=output)
model.summary()

# 设置tensorboard相关参数
# log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=(2,50))

#plot the model
#tf.keras.utils.plot_model(model, to_file='./1DCNNmodel.png', show_shapes=True, show_dtype=False,show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)


adam = Adam(learning_rate=0.00001,beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0,amsgrad=False)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#实际训练3456个，验证集384个
history = model.fit(X_train,y_train,epochs=200,batch_size=32,validation_split=0.1)
model.evaluate(X_test,y_test)
model.save('slim_Mobile_Lenet5.h5')

#训练过程可视化
fig = plt.figure()#新建一张图
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.plot(history.history['accuracy'],label='training acc')
plt.plot(history.history['val_accuracy'],label='val acc')
plt.title('model accuracy',fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.ylabel('accuracy',fontdict={'family':'Times New Roman', 'size':16})
plt.xlabel('epoch',fontdict={'family':'Times New Roman', 'size':16})
plt.legend(loc='lower right',prop={'family' : 'Times New Roman', 'size'   : 16})
fig.savefig('slim_Mobile_LeNet5_'+'acc.png')
fig = plt.figure()
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss',fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.ylabel('loss',fontdict={'family':'Times New Roman', 'size':16})
plt.xlabel('epoch',fontdict={'family':'Times New Roman', 'size':16})
plt.legend(loc='upper right',prop={'family' : 'Times New Roman', 'size'   : 16})
fig.savefig('slim_Mobile_LeNet5_'+'loss.png')
