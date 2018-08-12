import keras
from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.datasets import mnist

print('getting the data')
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(len(train_images))
print(train_images.dtype)

print(test_images.shape)
print(len(test_images))
print(test_images.dtype)

print("preparing the image data")
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

print("preparing the labels")
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print("defining the network")
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

print("compiling the network")
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print("training the network")
network.fit(train_images, train_labels, epochs=5, batch_size=128)

print("evaluating the network")
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print('test accuracy: ', test_accuracy)

