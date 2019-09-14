import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics as metri
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data_skeleton():
    train_data = np.empty(shape=(0, 218))
    test_data = np.empty(shape=(0, 218))

    hello = np.genfromtxt("./skeleton/Hello.csv", delimiter=',')
    call = np.genfromtxt("./skeleton/Call.csv", delimiter=',')
    stop = np.genfromtxt("./skeleton/Stop.csv", delimiter=',')
    pointing = np.genfromtxt("./skeleton/Pointing.csv", delimiter=',')
    coming = np.genfromtxt("./skeleton/Coming.csv", delimiter=',')
    going = np.genfromtxt("./skeleton/Going.csv", delimiter=',')

    test_file = np.genfromtxt("./skeleton/Test1.csv", delimiter=',')

    print('hello = ', hello.shape)
    print('call = ', call.shape)
    print('stop = ', stop.shape)
    print('pointing = ', pointing.shape)
    print('coming = ', coming.shape)
    print('going = ', going.shape)
    print('test_file = ', test_file.shape)

    training_percentage = 0.9

    train_data = np.concatenate((train_data, hello[0:int(training_percentage * len(hello))]), axis=0)
    train_data = np.concatenate((train_data, call[0:int(training_percentage * len(call))]), axis=0)
    train_data = np.concatenate((train_data, stop[0:int(training_percentage * len(stop))]), axis=0)
    train_data = np.concatenate((train_data, pointing[0:int(training_percentage * len(pointing))]), axis=0)
    train_data = np.concatenate((train_data, coming[0:int(training_percentage * len(coming))]), axis=0)
    train_data = np.concatenate((train_data, going[0:int(training_percentage * len(going))]), axis=0)

    test_data = np.concatenate((test_data, hello[int(training_percentage * len(hello)):]), axis=0)
    test_data = np.concatenate((test_data, call[int(training_percentage * len(call)):]), axis=0)
    test_data = np.concatenate((test_data, stop[int(training_percentage * len(stop)):]), axis=0)
    test_data = np.concatenate((test_data, pointing[int(training_percentage * len(pointing)):]), axis=0)
    test_data = np.concatenate((test_data, coming[int(training_percentage * len(coming)):]), axis=0)
    test_data = np.concatenate((test_data, going[int(training_percentage * len(going)):]), axis=0)

    train_data = np.delete(train_data, train_data.shape[1] - 1, axis=1)
    test_data = np.delete(test_data, test_data.shape[1] - 1, axis=1)
    test_file = np.delete(test_file, test_file.shape[1] - 1, axis=1)

    x_train = np.delete(train_data, 0, axis=1)
    x_test = np.delete(test_data, 0, axis=1)
    xtest_final = np.delete(test_file, 0, axis=1)

    y_train = train_data[:, 0]
    y_test = test_data[:, 0]
    ytest_final = test_file[:, 0]

    y_train = tf.keras.utils.to_categorical(y_train)
    y_train = np.delete(y_train, 0, axis=1)
    y_test = tf.keras.utils.to_categorical(y_test)
    y_test = np.delete(y_test, 0, axis=1)
    ytest_final = tf.keras.utils.to_categorical(ytest_final)
    ytest_final = np.delete(ytest_final, 0, axis=1)

    print('x_train = ', x_train.shape)

    print('x_test = ', x_test.shape)

    print('xtest_final = ', xtest_final.shape)

    print('y_train = ', y_train.shape)

    print('y_test = ', y_test.shape)

    print('ytest_final = ', ytest_final.shape)

    return x_train, x_test, y_train, y_test, xtest_final, ytest_final


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
                         , metrics=['accuracy',
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
        # i = i.reshape(1, x, y, 1)
        out = loaded_model.predict(instance)
        real = np.argmax(yTesting[i])
        if real == np.argmax(out, axis=1):
            true += 1

        print(np.argmax(out, axis=1), "====", real)
    true = true / len(xTesting)
    print(true)
    loaded_model.summary()
#3,-0.196735,-0.243866,-0.290222,0.254872,0.0116032,3.13793,0.234211,0.068166,0.393083,0.254872,0.0116032,3.13793,0.109276,-0.117753,2.40722,1.82566,-0.00646996,1.5811,-0.20968,-0.0468809,-0.0746755,-1.20253,-0.229589,1.60908,0.347283,0.0162924,0.122249,2.08803,0.30377,1.96071,-0.00632181,0.193577,0.35131,-0.012313,3.75808e-05,0.00707752,-0.00235293,0.0956325,0.171021,-0.0123262,4.13223e-05,0.0070754,0.156359,0.0937266,0.174268,1.00324,0.181261,-0.421229,-0.161059,0.0975321,0.167773,-1.18842,-0.0306463,1.38869,0.00170982,0.00462921,-0.000345977,0.246481,0.0131422,3.13726,0.0360305,-8.68735e-05,0.0133957,0.246481,0.0131422,3.13726,-0.00162445,-0.000534216,-0.00128735,1.81727,-0.00740856,1.58249,0.000552169,0.00298297,0.00107867,-1.20271,-0.23866,1.61227,0.00482852,-0.00416868,0.00724483,2.00747,0.288222,1.8442,-0.000761344,-0.00693476,0.00325694,-0.0170242,6.24767e-05,0.0070777,-0.000362804,-0.00287487,0.00157452,-0.0170156,6.08035e-05,0.00707712,-0.000375181,-0.00316811,0.00200744,1.08861,0.197313,-0.39199,-0.000354846,-0.00256908,0.00114451,-1.18842,-0.0307084,1.38842,-0.00105992,0.00171902,-0.00457992,0.238618,0.0178564,3.13272,0.0405451,-0.00756701,0.0101231,0.238618,0.0178564,3.13272,0.000655637,-0.000228132,0.00100317,1.8094,-0.0128461,1.58605,0.000256365,0.000758103,-0.00364041,-1.20224,-0.249444,1.60397,0.00835941,-0.00427938,0.00521999,1.93261,0.293768,1.71909,-0.00254248,-0.00563932,0.00335774,-0.0188312,6.01667e-05,0.00590138,-0.00133146,-0.00268277,0.00139178,-0.0188256,5.67178e-05,0.00589821,-0.00140113,-0.00376298,0.00309867,1.14741,0.190578,-0.371498,-0.0012617,-0.00161011,-0.000318397,-1.18842,-0.0349209,1.3883,0,0,0,0.238618,0.0178564,3.13272,0,0,0,0.238618,0.0178564,3.13272,0,0,0,1.8094,-0.0128461,1.58605,0,0,0,-1.20224,-0.249444,1.60397,0,0,0,1.93261,0.293768,1.71909,0,0,0,-0.0188312,6.01667e-05,0.00590138,0,0,0,-0.0188256,5.67178e-05,0.00589821,0,0,0,1.14741,0.190578,-0.371498,0,0,0,-1.18842,-0.0349209,1.3883,


instance = -0.196735,-0.243866,-0.290222,0.254872,0.0116032,3.13793,0.234211,0.068166,0.393083,0.254872,0.0116032,3.13793,0.109276,-0.117753,2.40722,1.82566,-0.00646996,1.5811,-0.20968,-0.0468809,-0.0746755,-1.20253,-0.229589,1.60908,0.347283,0.0162924,0.122249,2.08803,0.30377,1.96071,-0.00632181,0.193577,0.35131,-0.012313,3.75808e-05,0.00707752,-0.00235293,0.0956325,0.171021,-0.0123262,4.13223e-05,0.0070754,0.156359,0.0937266,0.174268,1.00324,0.181261,-0.421229,-0.161059,0.0975321,0.167773,-1.18842,-0.0306463,1.38869,0.00170982,0.00462921,-0.000345977,0.246481,0.0131422,3.13726,0.0360305,-8.68735e-05,0.0133957,0.246481,0.0131422,3.13726,-0.00162445,-0.000534216,-0.00128735,1.81727,-0.00740856,1.58249,0.000552169,0.00298297,0.00107867,-1.20271,-0.23866,1.61227,0.00482852,-0.00416868,0.00724483,2.00747,0.288222,1.8442,-0.000761344,-0.00693476,0.00325694,-0.0170242,6.24767e-05,0.0070777,-0.000362804,-0.00287487,0.00157452,-0.0170156,6.08035e-05,0.00707712,-0.000375181,-0.00316811,0.00200744,1.08861,0.197313,-0.39199,-0.000354846,-0.00256908,0.00114451,-1.18842,-0.0307084,1.38842,-0.00105992,0.00171902,-0.00457992,0.238618,0.0178564,3.13272,0.0405451,-0.00756701,0.0101231,0.238618,0.0178564,3.13272,0.000655637,-0.000228132,0.00100317,1.8094,-0.0128461,1.58605,0.000256365,0.000758103,-0.00364041,-1.20224,-0.249444,1.60397,0.00835941,-0.00427938,0.00521999,1.93261,0.293768,1.71909,-0.00254248,-0.00563932,0.00335774,-0.0188312,6.01667e-05,0.00590138,-0.00133146,-0.00268277,0.00139178,-0.0188256,5.67178e-05,0.00589821,-0.00140113,-0.00376298,0.00309867,1.14741,0.190578,-0.371498,-0.0012617,-0.00161011,-0.000318397,-1.18842,-0.0349209,1.3883,0,0,0,0.238618,0.0178564,3.13272,0,0,0,0.238618,0.0178564,3.13272,0,0,0,1.8094,-0.0128461,1.58605,0,0,0,-1.20224,-0.249444,1.60397,0,0,0,1.93261,0.293768,1.71909,0,0,0,-0.0188312,6.01667e-05,0.00590138,0,0,0,-0.0188256,5.67178e-05,0.00589821,0,0,0,1.14741,0.190578,-0.371498,0,0,0,-1.18842,-0.0349209,1.3883,

instance = instance.reshape(1, x, y, 1)
# i = i.reshape(1, x, y, 1)
out = loaded_model.predict(instance)
real = np.argmax(yTesting[i])
print('real =',3)
print('out=',np.argmax(out, axis=1))