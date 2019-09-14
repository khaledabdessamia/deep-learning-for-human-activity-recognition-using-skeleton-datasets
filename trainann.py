import tensorflow as tf
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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


def run_ann(x_train, x_test, y_train, y_test, xTesting, yTesting, batch_size, epochs):

    model = tf.keras.models.Sequential([
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(y_train.shape[1], activation=tf.nn.softmax)
    ])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  # optimizer=tf.train.AdadeltaOptimizer(),
                  metrics=['accuracy'])

    #   tf.keras.optimizers.Adadelta(),

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_train, y_train)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    score = model.evaluate(x_test, y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save the model
    # serialize model to JSON
    model_json = model.to_json()
    with open("modelann.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("modelann.h5")
    print("Saved model to disk")


def main():
    x_traing, x_testg, y_traing, y_testg, xtestFinal, ytestFinal = load_data_skeleton()

    batch_size = 4

    epochs = 7

    run_ann(x_traing, x_testg, y_traing, y_testg, x_testg, y_testg, batch_size, epochs)


if __name__ == "__main__":
    main()
