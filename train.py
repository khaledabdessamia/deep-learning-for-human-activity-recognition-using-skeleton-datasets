import tensorflow as tf
import os
import numpy as np

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


def run_test(x_train, x_test, y_train, y_test, batch_size, epochs):
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(y_train.shape[1], activation=tf.nn.softmax)
    ])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  # optimizer=tf.train.AdadeltaOptimizer(),
                  metrics=['accuracy'])

    #   tf.keras.optimizers.Adadelta(),

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save the model
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def run_cnn(x_train, x_test, y_train, y_test, xTesting, yTesting, x, y, batch_size, epochs):
    x_train = x_train.reshape(x_train.shape[0], x, y, 1)
    x_test = x_test.reshape(x_test.shape[0], x, y, 1)
    xTesting = xTesting.reshape(xTesting.shape[0], x, y, 1)
    input_shape = (x, y, 1)

    # convert the data to the right type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    xTesting = xTesting.astype('float32')
    # x_train /= 255
    # x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(xTesting.shape[0], 'testFinal samples')

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1),
                                     activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(512, (4, 4), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(400, activation='relu'))
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    score = model.evaluate(xTesting, yTesting)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save the model
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def run_cnnLstm(x_train, x_test, y_train, y_test, x, y, batch_size, epochs):
    x_train = x_train.reshape(x_train.shape[0], x, y, 1)
    x_test = x_test.reshape(x_test.shape[0], x, y, 1)
    input_shape = (x, y, 1)

    # convert the data to the right type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    #######################################################################
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(20000, 32, input_length=100))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                                     activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(128, (4, 4), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.35))
    model.add(tf.keras.layers.Conv2D(128, (4, 4), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.GRU(50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.45))
    model.add(tf.keras.layers.Dense(6, activation='sigmoid'))
    # model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save the model
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def run_rnn(x_train, x_test, y_train, y_test, batch_size, epochs):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_train.shape[1], 1)
    # input_shape = (img_x, img_y, 1)

    # convert the data to the right type
    # x_train = x_train.astype('int64')
    # x_test = x_test.astype('int64')
    # x_train /= 255
    # x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = tf.keras.models.Sequential()

    # First, let's define a RNN Cell, as a layer subclass.

    class MinimalRNNCell(tf.keras.layers.Layer):

        def __init__(self, units, **kwargs):
            self.units = units
            self.state_size = units
            super(MinimalRNNCell, self).__init__(**kwargs)

        def build(self, input_shape):
            self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                          initializer='uniform',
                                          name='kernel')
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='uniform',
                name='recurrent_kernel')
            self.built = True

        def call(self, inputs, states):
            prev_output = states[0]
            h = tf.keras.backend.dot(inputs, self.kernel)
            output = h + tf.keras.backend.dot(prev_output, self.recurrent_kernel)
            return output, [output]

    # Let's use this cell in a RNN layer:

    cell = MinimalRNNCell(32)
    x = tf.keras.Input((None,))
    layer = tf.keras.layers.RNN(cell, input_shape=(None, 144))(x)
    # y = layer(x)

    model.add(tf.keras.layers.RNN(cell))

    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid'))

    model.compile(loss='car',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save the model
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def run_lstm(x_train, x_test, y_train, y_test, xTesting, yTesting, x, y, batch_size, epochs):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    # input_shape = (img_x, img_y, 1)
    xTesting = xTesting.reshape(xTesting.shape[0], xTesting.shape[1], 1)
    # convert the data to the right type
    # x_train = x_train.astype('int64')
    # x_test = x_test.astype('int64')
    # x_train /= 255
    # x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # input1=tf.keras.Input(shape=(None,144))
    # #model.add(tf.keras.layers.Embedding(4096, output_dim=256))
    # lstm1=tf.keras.layers.LSTM(500,)(input1)
    # #model.add(tf.keras.layers.Dropout(0.5))
    #
    # dense1=tf.keras.layers.Dense(18, activation='sigmoid')(lstm1)
    #
    # model = tf.keras.models.Model(inputs=input1,outputs=dense1)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(10))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500, activation='relu'))

    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adagrad(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=20, epochs=6, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    score = model.evaluate(xTesting, yTesting)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save the model
    # serialize model to JSON
    model_json = model.to_json()
    with open("modellstm.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("modellstm.h5")
    print("Saved model to disk")


def run_lstm2(x_train, x_test, y_train, y_test, batch_size, epochs):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    ### model = tf.keras.models.Sequential()

    # model.add(tf.keras.layers.LSTM(50))
    # model.add(tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid'))

    # model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.RMSprop(), metrics=['accuracy'])

    embed_dim = 128
    lstm_out = 300
    batch_size = 32

    ##Buidling the LSTM network

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(2500, embed_dim, input_length=x_train.shape, dropout=0.1))
    model.add(tf.keras.layers.LSTM(lstm_out, dropout_U=0.1, dropout_W=0.1))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save the model
    # serialize model to JSON
    model_json = model.to_json()
    with open("modellstm.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("modellstm.h5")
    print("Saved model to disk")


##****************************************************************************

def main():

    x_traing, x_testg, y_traing, y_testg, xTesting, yTesting = load_data_skeleton()

    batch_size = 5

    epochs = 18

    #run_test(x_traing, x_testg, y_traing, y_testg,batch_size,epochs)

    x = 12
    y = 18

    # run_lstm(x_traing, x_testg, y_traing,y_testg,xTesting,yTesting,x,y,batch_size,epochs)
    # run_cnnLstm(x_traing, x_testg, y_traing, y_testg,x,y,batch_size,epochs)

    # run_rnn(x_traing, x_testg, y_traing, y_testg,batch_size,epochs)
    # run_lstm(x_traing, x_testg, y_traing, y_testg, batch_size, epochs)

    # run_lstm2(x_traing, x_testg, y_traing, y_testg, batch_size, epochs)

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
    loaded_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                         metrics=['accuracy'])

    score = loaded_model.evaluate(x_testg, y_testg)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    score = loaded_model.evaluate(xTesting, yTesting)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])





if __name__ == "__main__":
    main()
