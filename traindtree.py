from sklearn import tree
import numpy as np
import tensorflow as tf


def load_data_skeleton():
    train_data = np.empty(shape=(0, 218))
    test_data = np.empty(shape=(0, 218))
    test_final_data = np.empty(shape=(0, 218))

    hello = np.genfromtxt("./skeleton/Hello.csv", delimiter=',')
    call = np.genfromtxt("./skeleton/Call.csv", delimiter=',')
    stop = np.genfromtxt("./skeleton/Stop.csv", delimiter=',')
    pointing = np.genfromtxt("./skeleton/Pointing.csv", delimiter=',')
    coming = np.genfromtxt("./skeleton/Coming.csv", delimiter=',')
    going = np.genfromtxt("./skeleton/Going.csv", delimiter=',')

    TestFile = np.genfromtxt("./skeleton/Test1.csv", delimiter=',')

    print('hello = ', hello.shape)
    print('call = ', call.shape)
    print('stop = ', stop.shape)
    print('pointing = ', pointing.shape)
    print('coming = ', coming.shape)
    print('going = ', going.shape)
    print('TestFile = ', TestFile.shape)

    training_pourcentage = 0.9

    train_data = np.concatenate((train_data, hello[0:int(training_pourcentage * len(hello))]), axis=0)
    train_data = np.concatenate((train_data, call[0:int(training_pourcentage * len(call))]), axis=0)
    train_data = np.concatenate((train_data, stop[0:int(training_pourcentage * len(stop))]), axis=0)
    train_data = np.concatenate((train_data, pointing[0:int(training_pourcentage * len(pointing))]), axis=0)
    train_data = np.concatenate((train_data, coming[0:int(training_pourcentage * len(coming))]), axis=0)
    train_data = np.concatenate((train_data, going[0:int(training_pourcentage * len(going))]), axis=0)

    test_data = np.concatenate((test_data, hello[int(training_pourcentage * len(hello)):]), axis=0)
    test_data = np.concatenate((test_data, call[int(training_pourcentage * len(call)):]), axis=0)
    test_data = np.concatenate((test_data, stop[int(training_pourcentage * len(stop)):]), axis=0)
    test_data = np.concatenate((test_data, pointing[int(training_pourcentage * len(pointing)):]), axis=0)
    test_data = np.concatenate((test_data, coming[int(training_pourcentage * len(coming)):]), axis=0)
    test_data = np.concatenate((test_data, going[int(training_pourcentage * len(going)):]), axis=0)

    train_data = np.delete(train_data, train_data.shape[1] - 1, axis=1)
    test_data = np.delete(test_data, test_data.shape[1] - 1, axis=1)
    TestFile = np.delete(TestFile, TestFile.shape[1] - 1, axis=1)

    x_train = np.delete(train_data, 0, axis=1)
    x_test = np.delete(test_data, 0, axis=1)
    xtestFinal = np.delete(TestFile, 0, axis=1)

    y_train = train_data[:, 0]
    y_test = test_data[:, 0]
    ytestFinal = TestFile[:, 0]

    y_train = tf.keras.utils.to_categorical(y_train)
    y_train = np.delete(y_train, 0, axis=1)
    y_test = tf.keras.utils.to_categorical(y_test)
    y_test = np.delete(y_test, 0, axis=1)
    ytestFinal = tf.keras.utils.to_categorical(ytestFinal)
    ytestFinal = np.delete(ytestFinal, 0, axis=1)

    print('x_train = ', x_train.shape)

    print('x_test = ', x_test.shape)

    print('xtestFinal = ', xtestFinal.shape)

    print('y_train = ', y_train.shape)

    print('y_test = ', y_test.shape)

    print('ytestFinal = ', ytestFinal.shape)

    return x_train, x_test, y_train, y_test, xtestFinal, ytestFinal


X, x_testg, Y, y_testg, xTesting, yTesting = load_data_skeleton()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

true = 0
for i in range(len(x_testg)):
    if np.argmax(clf.predict([x_testg[i]]), axis=1) == np.argmax(y_testg[i], axis=0):
        true += 1
true = true / len(x_testg)
print(true)

true = 0
for i in range(len(xTesting)):
    if np.argmax(clf.predict([xTesting[i]]), axis=1) == np.argmax(yTesting[i], axis=0):
        true += 1
true = true / len(xTesting)
print(true)

xx=np.concatenate((x_testg, xTesting[:]), axis=0)
yy=np.concatenate((y_testg, yTesting[:]), axis=0)

true = 0
for i in range(len(xx)):
    if np.argmax(clf.predict([xx[i]]), axis=1) == np.argmax(yy[i], axis=0):
        true += 1
true = true / len(xx)
print(true)