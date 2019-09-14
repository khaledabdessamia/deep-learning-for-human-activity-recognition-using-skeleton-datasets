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
        element = "./cad60/%s" % element
        if element.endswith('.txt'):
            data = np.genfromtxt(element, delimiter=',')

            x_train = np.concatenate((x_train, data[0:int(0.7 * len(data))]), axis=0)

            x_test = np.concatenate((x_test, data[int(0.7 * len(data)):]), axis=0)

            element = element.replace('./cad60/', '').replace('.txt', '')
            # s = element.split(".")
            z = np.zeros(shape=18)
            # z[num.index(int(s[0]))] = int(1)
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


def run_cnncad(x_train, x_test, y_train, y_test, xTesting, yTesting, x, y, batch_size, epochs):
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
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
                                     activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (4, 4), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
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
    with open("modelcad2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("modelcad2.h5")
    print("Saved model to disk")


def main():
    x_traing, x_testg, y_traing, y_testg = load_data_cad60()

    batch_size = 1000

    epochs = 6

    x = 12
    y = 12

    run_cnncad(x_traing, x_testg, y_traing, y_testg, x_testg, y_testg, x, y, batch_size, epochs)


if __name__ == "__main__":
    main()
