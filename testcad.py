import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics as metri
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


if __name__ == "__main__":
    x = 12
    y = 12

    x_traing, x_testg, y_traing, y_testg = load_data_cad60()

    x_traing = x_traing.reshape(x_traing.shape[0], x, y, 1)

    x_testg = x_testg.reshape(x_testg.shape[0], x, y, 1)

    json_file = open('modelcad.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load woeights into new model
    loaded_model.load_weights("modelcad.h5")
    print("Loaded Model from disk")

    # compile and evaluate loaded model
    loaded_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                         metrics=['accuracy'])
    score0 = loaded_model.evaluate(x_traing, y_traing)

    print('Test loss:', score0[0])
    print('Test accuracy:', score0[1])

    score1 = loaded_model.evaluate(x_testg, y_testg)

    print('Test loss:', score1[0])
    print('Test accuracy:', score1[1])

    print()
    print('les test sur la base de donnes CAD :')
    print('Test loss on the training data :', score0[0])
    print('Test accuracy on the training data :', score0[1])

    print('Test loss on the testing data :', score1[0])
    print('Test accuracy on the testin data :', score1[1])
