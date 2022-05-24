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


################
###正解率の評価###
################

# テスト用データセットに対するモデルの性能を比較する
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

###予測する###
# モデルを使って画像の分類予測を行う
predictions = model.predict(test_images)
print("学習結果[予測]："+str(np.argmax(predictions[0]))) # predictions[0]最大値
print("学習結果[正解]："+str(test_labels[0]))

###正解の評価グラフ表示###
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
# X個のテスト画像、予測されたラベル、正解ラベルを表示します。
# 正しい予測は青で、間違った予測は赤で表示しています。
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
