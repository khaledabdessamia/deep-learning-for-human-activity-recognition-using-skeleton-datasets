import tensorflow as tf
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_data_cad60():
    labels = np.genfromtxt("./cad60/activityLabel.csv", delimiter=',', dtype=None, encoding=None)

    num = []
    label = []
    for e in labels:
        num.append(e[0])
        label.append(e[1])

    x_train = np.empty(shape=(0, 172))
    x_test = np.empty(shape=(0, 172))
    y_train = np.empty(shape=(0, 18))
    y_test = np.empty(shape=(0, 18))

    for element in os.listdir('./cad60/.'):
        element= "./cad60/%s" %(element)
        if element.endswith('.txt'):
            data = np.genfromtxt(element, delimiter=',')

            x_train = np.concatenate((x_train, data[0:int(0.7 * len(data))]), axis=0)

            x_test = np.concatenate((x_test, data[int(0.7 * len(data)):]), axis=0)

            element = element.replace('./cad60/','').replace('.txt','')
            #s = element.split(".")
            z = np.zeros(shape=18)
            #z[num.index(int(s[0]))] = int(1)
            z[num.index(int(element))] = int(1)
            y_train = np.concatenate((y_train, [z for i in range(int(0.7 * len(data)))]), axis=0)

            y_test = np.concatenate((y_test, [z for i in range(len(data) - int(0.7 * len(data)))]), axis=0)

    # first column frame
    x_train = np.delete(x_train, 0, 1)

    x_train = np.delete(x_train, 9, 1)

    x_train = np.delete(x_train, 12, 1)

    x_train = np.delete(x_train, 21, 1)

    x_train = np.delete(x_train, 24, 1)

    x_train = np.delete(x_train, 33, 1)

    x_train = np.delete(x_train, 36, 1)

    x_train = np.delete(x_train, 45, 1)

    x_train = np.delete(x_train, 48, 1)

    x_train = np.delete(x_train, 57, 1)

    x_train = np.delete(x_train, 60, 1)

    x_train = np.delete(x_train, 69, 1)

    x_train = np.delete(x_train, 72, 1)

    x_train = np.delete(x_train, 81, 1)

    x_train = np.delete(x_train, 84, 1)

    x_train = np.delete(x_train, 93, 1)

    x_train = np.delete(x_train, 96, 1)

    x_train = np.delete(x_train, 105, 1)

    x_train = np.delete(x_train, 108, 1)

    x_train = np.delete(x_train, 127, 1)

    x_train = np.delete(x_train, 120, 1)

    x_train = np.delete(x_train, 129, 1)

    x_train = np.delete(x_train, 132, 1)

    x_train = np.delete(x_train, 135, 1)

    x_train = np.delete(x_train, 138, 1)

    x_train = np.delete(x_train, 141, 1)

    x_train = np.delete(x_train, 144, 1)

    x_train = np.delete(x_train, x_train.shape[1] - 1, 1)

    # first column frame
    x_test = np.delete(x_test, 0, 1)

    x_test = np.delete(x_test, 9, 1)

    x_test = np.delete(x_test, 12, 1)

    x_test = np.delete(x_test, 21, 1)

    x_test = np.delete(x_test, 24, 1)

    x_test = np.delete(x_test, 33, 1)

    x_test = np.delete(x_test, 36, 1)

    x_test = np.delete(x_test, 45, 1)

    x_test = np.delete(x_test, 48, 1)

    x_test = np.delete(x_test, 57, 1)

    x_test = np.delete(x_test, 60, 1)

    x_test = np.delete(x_test, 69, 1)

    x_test = np.delete(x_test, 72, 1)

    x_test = np.delete(x_test, 81, 1)

    x_test = np.delete(x_test, 84, 1)

    x_test = np.delete(x_test, 93, 1)

    x_test = np.delete(x_test, 96, 1)

    x_test = np.delete(x_test, 105, 1)

    x_test = np.delete(x_test, 108, 1)

    x_test = np.delete(x_test, 127, 1)

    x_test = np.delete(x_test, 120, 1)

    x_test = np.delete(x_test, 129, 1)

    x_test = np.delete(x_test, 132, 1)

    x_test = np.delete(x_test, 135, 1)

    x_test = np.delete(x_test, 138, 1)

    x_test = np.delete(x_test, 141, 1)

    x_test = np.delete(x_test, 144, 1)

    x_test = np.delete(x_test, x_test.shape[1] - 1, 1)

    print('x_train = ', x_train.shape)

    print('x_test = ', x_test.shape)

    print('y_train = ', y_train.shape)

    print('y_test = ', y_test.shape)

    return x_train, x_test, y_train, y_test


def load_data_skeleton():
    train_data = np.empty(shape=(0, 218))
    test_data = np.empty(shape=(0, 218))

    hello = np.genfromtxt("./skeleton/Hello.csv", delimiter=',')
    call = np.genfromtxt("./skeleton/Call.csv", delimiter=',')
    stop = np.genfromtxt("./skeleton/Stop.csv", delimiter=',')
    pointing = np.genfromtxt("./skeleton/Pointing.csv", delimiter=',')
    coming = np.genfromtxt("./skeleton/Coming.csv", delimiter=',')
    # going = np.genfromtxt("./skeleton/Going.csv", delimiter=',')

    print('hello = ', hello.shape)
    print('call = ', call.shape)
    print('stop = ', stop.shape)
    print('pointing = ', pointing.shape)
    print('coming = ', coming.shape)
    # print('going = ', going.shape)

    training_pourcentage = 0.7

    train_data = np.concatenate((train_data, hello[0:int(training_pourcentage * len(hello))]), axis=0)
    train_data = np.concatenate((train_data, call[0:int(training_pourcentage * len(call))]), axis=0)
    train_data = np.concatenate((train_data, stop[0:int(training_pourcentage * len(stop))]), axis=0)
    train_data = np.concatenate((train_data, pointing[0:int(training_pourcentage * len(pointing))]), axis=0)
    train_data = np.concatenate((train_data, coming[0:int(training_pourcentage * len(coming))]), axis=0)
    #    train_data = np.concatenate((train_data, going[0:int(training_pourcentage * len(going))]), axis=0)

    test_data = np.concatenate((test_data, hello[int(training_pourcentage * len(hello)):]), axis=0)
    test_data = np.concatenate((test_data, call[int(training_pourcentage * len(call)):]), axis=0)
    test_data = np.concatenate((test_data, stop[int(training_pourcentage * len(stop)):]), axis=0)
    test_data = np.concatenate((test_data, pointing[int(training_pourcentage * len(pointing)):]), axis=0)
    test_data = np.concatenate((test_data, coming[int(training_pourcentage * len(coming)):]), axis=0)
    #    test_data = np.concatenate((test_data, going[int(training_pourcentage * len(going)):]), axis=0)

    train_data = np.delete(train_data, train_data.shape[1] - 1, axis=1)

    test_data = np.delete(test_data, test_data.shape[1] - 1, axis=1)

    x_train = np.delete(train_data, 0, axis=1)
    x_test = np.delete(test_data, 0, axis=1)

    y_train = train_data[:, 0]
    y_test = test_data[:, 0]

    y_train = tf.keras.utils.to_categorical(y_train)
    y_train = np.delete(y_train, 0, axis=1)
    y_test = tf.keras.utils.to_categorical(y_test)
    y_test = np.delete(y_test, 0, axis=1)

    print('x_train = ', x_train.shape)

    print('x_test = ', x_test.shape)

    print('y_train = ', y_train.shape)

    print('y_test = ', y_test.shape)

    return x_train, x_test, y_train, y_test


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
                  #optimizer=tf.train.AdadeltaOptimizer(),
                  metrics=['accuracy'])

    #   tf.keras.optimizers.Adadelta(),

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def run_cnn(x_train, x_test, y_train, y_test, x, y, batch_size, epochs):
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

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                     activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (4, 4), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    #   tf.keras.optimizers.Adadelta(),

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


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


def run_lstm(x_train, x_test, y_train, y_test, batch_size, epochs):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    # input_shape = (img_x, img_y, 1)

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

    model.add(tf.keras.layers.LSTM(500))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.RMSprop(),
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


if __name__ == "__main__":
    #x_traing, x_testg, y_traing, y_testg = load_data_cad60()
    x_traing, x_testg, y_traing, y_testg = load_data_skeleton()

    batch_size = 5

    epochs = 20

    run_test(x_traing, x_testg, y_traing, y_testg,batch_size,epochs)

    x = 12
    y = 18
    #run_cnn(x_traing, x_testg, y_traing, y_testg,x,y,batch_size,epochs)

    #run_rnn(x_traing, x_testg, y_traing, y_testg,batch_size,epochs)

    #run_lstm(x_traing, x_testg, y_traing, y_testg, batch_size, epochs)
