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
