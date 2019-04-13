import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics as metri


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

    TestFile = np.genfromtxt("./skeleton/Filles_dataset.csv", delimiter=',')

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


if __name__ == "__main__":

    x = 12
    y = 18

    x_traing, x_testg, y_traing, y_testg, xTesting, yTesting = load_data_skeleton()

    x_traing = x_traing.reshape(x_traing.shape[0], x, y, 1)

    x_testg = x_testg.reshape(x_testg.shape[0], x, y, 1)

    xTesting = xTesting.reshape(xTesting.shape[0], x, y, 1)

    json_file = open('modelCNN986-0.06.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load woeights into new model
    loaded_model.load_weights("modelCNN986-0.06.h5")
    print("Loaded Model from disk")

    # compile and evaluate loaded model
    loaded_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam()
                         ,metrics=['accuracy',
                                   metri.sparse_top_k_categorical_accuracy

                                   ])

    score = loaded_model.evaluate(x_traing, y_traing)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print(score)
    score = loaded_model.evaluate(x_testg, y_testg)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    score = loaded_model.evaluate(xTesting, yTesting)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # indice = 2
    # instance = x_testg[indice]
    # instance = instance.reshape(1, x, y, 1)
    #
    # out = loaded_model.predict(instance)
    # print(out)
    # print(np.argmax(out, axis=1))
    # print(np.argmax(y_testg[indice]))
    true = 0
    for i in range(len(xTesting)):
        instance = xTesting[i]
        instance = instance.reshape(1, x, y, 1)
        #i = i.reshape(1, x, y, 1)
        out = loaded_model.predict(instance)
        real = np.argmax(yTesting[i])
        if(real == np.argmax(out, axis=1)):
            true += 1

        print(np.argmax(out, axis=1),"====",real)
    true = true / len(xTesting)
    print(true)