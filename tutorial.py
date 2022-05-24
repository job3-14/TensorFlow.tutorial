# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

#######################################
### ファッションMNISTデータセットのロード###
######################################
# mnistをロード
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# train_images , train_labels 訓練データセット
# test_images , test_labels テスト用データセット
# 画像は28x28 それぞれのピクセルは０から255の整数

# Label0から9にクラス名を作成
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat' ,'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


################
###データの観察###
################

# 訓練用データセットには28×28ピクセルの画像が60,000枚含まれる
#print(train_images.shape) # (60000, 28, 28)

# 訓練用データセットには60,000個のラベルが含まれる
#print(len(train_labels))

# ラベルは0から9である
#print(train_labels)

# 訓練用データセットには28×28ピクセルの画像が10,000枚含まれる
#print(test_images.shape)

# テスト用データセットには10,000個のラベルが含まれる
#print(len(test_labels))


##################
###データの前処理###
#################

#train_images[0]をグラフ化
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''

# ニューラルネットワークにデータを投入する前に値を0から1に変換する
# 画素である255で割る
train_images = train_images / 255.0
test_images = test_images / 255.0

# 訓練用データセットの初めの25枚をグラフ表示する
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

################
###モデルの構築###
###############

###層の設定###
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

###モデルのコンパイル###
model.compile(optimizer='adam',  # オプティマイザ
              loss='sparse_categorical_crossentropy', # 損失関数
              metrics=['accuracy']) # メトリクス

################
###モデルの訓練###
###############
model.fit(train_images, train_labels, epochs=5)
# モデルにtrain_images, train_labelsの配列を投入
