from keras.layers import Input, Dense, Activation
from keras.models import Model

from keras.datasets import mnist
from keras.optimizers import Adadelta
from keras.utils import to_categorical
import numpy as np

encoding_dim = 256

input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='relu')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

autoencoder.compile(optimizer=Adadelta(lr=10), loss='binary_crossentropy')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = 1*(x_train.astype('float32') >= 128)
x_test = 1*(x_test.astype('float32') >= 128)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

try:
	RBM = np.load("RBM_weights.npy")
	W1, b1 = autoencoder.layers[1].get_weights()
	autoencoder.layers[1].set_weights([RBM.T, b1])
	_, b2 = autoencoder.layers[2].get_weights()
	autoencoder.layers[2].set_weights([RBM, b2])
	print("RBM weights used to initialize Autoencoder")
except Exception, e:
	print(e)
	pass

autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, validation_split=0.2)
print("Reconstruction loss on test data is ", autoencoder.evaluate(x_test, x_test))

# Freeze encoder layers
for layer in encoder.layers:
	layer.trainable = False

x = Activation('relu')(encoder.output)
x = Dense(128, activation='sigmoid')(x)
output = Dense(10, activation='softmax')(x)

classifier = Model(encoder.input, output)
classifier.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(x_train, y_train, epochs=10, batch_size=256, validation_split=0.2)
print("Classification accuracy on test data is ", classifier.evaluate(x_test, y_test)[1])
