__author__ = 'yuhongliang324'

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import cPickle, numpy

data_path = '/usr0/home/hongliay/datasets/svhn_hid'
batch_size = 100
epochs = 10

print 'Loading data ... '
reader = open(data_path)
X, Y = cPickle.load(reader)
reader.close()
print 'Data loaded'

X_all = numpy.concatenate(X, axis=1)
Y_all = numpy.concatenate(Y, axis=0)

n_train, n_test = 60000, 10000

X_train = X_all[:, :n_train, :]
Y_train = Y_all[:n_train, :]
X_test = X_all[:, n_train: n_train + n_test, :]
Y_test = Y_all[n_train: n_train + n_test, :]

T = X_train.shape[0]

for i in xrange(T):
    x_train = X_train[i]
    y_train = Y_train
    x_test = X_test[i]
    y_test = Y_test
    print x_train.shape, y_train.shape, x_test.shape, y_test.shape
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(256,)))
    model.add(Dropout(0.2))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    print 'Compiling ... '
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    print 'Compilation done ... '
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

