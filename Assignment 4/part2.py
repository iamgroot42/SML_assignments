from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import Adadelta, SGD


def loadData(dataPath):
	return (X_train, Y_train), (X_test, Y_test)


def getNetwork(input_shape, d1, d2, output_shape, lr, act, d3=None)
	model = Sequential()
	model.add(Input(input_shape))
	model.add(Dense(d1))
	model.add(Activation(act))
	model.add(Dense(d2))
	if d3:
		model.add(Dense(d3))
		model.add(Activation(act))
	model.add(Dense(output_shape))
	model.add(Activation('softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=Adadelta(lr),
		metrics=['accuracy'])
	return model


def getSmallNetwork(input_shape, h, output_shape, lr):
	model = Sequential()
	model.add(Input(input_shape))
	model.add(Dense(h))
	model.add(Activation('relu'))
	model.add(Dense(otuput_shape))
	 model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=Adadelta(lr),
                metrics=['accuracy'])
	return model


if __name__ == "__main__":
	# Part 1
	m1 = getNetwork((256,), 128, 64, 47, 0.1, 'linear')
	# Part 2
	m2 = getNetwork((256,), 256, 128, 47, 0.1, 'linear')
	# Part 3
	m3 = getNetwork((256,), 128, 64, 47, 0.1, 'linear', 128)
	# Part 4
	m4 = getNetwork((256,), 128, 64, 47, 0.1, 'linear')
	# Part 5
	m5_1 = getNetwork((256,), 128, 64, 47, 0.1, 'sigmoid')
	m5_2 = getNetwork((256,), 128, 64, 47, 0.1, 'relu')
	# Part 6
	# Use best network so far, compress into 2 layer network using distillation
	m6 = getSmallNetwork((256,), 128, 47, 0.1)
