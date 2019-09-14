import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics as metri
import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

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

    test_file = np.genfromtxt("./skeleton/Test11.csv", delimiter=',')
    test_file2 = np.genfromtxt("./skeleton/Test2.csv", delimiter=',')

    print('hello = ', hello.shape)
    print('call = ', call.shape)
    print('stop = ', stop.shape)
    print('pointing = ', pointing.shape)
    print('coming = ', coming.shape)
    print('going = ', going.shape)
    print('test_file = ', test_file.shape)
    print('test_file2 = ', test_file2.shape)

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
    test_file2 = np.delete(test_file2, test_file2.shape[1] - 1, axis=1)

    x_train = np.delete(train_data, 0, axis=1)
    x_test = np.delete(test_data, 0, axis=1)
    xtest_final = np.delete(test_file, 0, axis=1)
    xtest_final2 = np.delete(test_file2, 0, axis=1)

    y_train = train_data[:, 0]
    y_test = test_data[:, 0]
    ytest_final = test_file[:, 0]
    ytest_final2 = test_file2[:, 0]

    y_train = tf.keras.utils.to_categorical(y_train)
    y_train = np.delete(y_train, 0, axis=1)
    y_test = tf.keras.utils.to_categorical(y_test)
    y_test = np.delete(y_test, 0, axis=1)
    ytest_final = tf.keras.utils.to_categorical(ytest_final)
    ytest_final = np.delete(ytest_final, 0, axis=1)
    ytest_final2 = tf.keras.utils.to_categorical(ytest_final2)
    ytest_final2 = np.delete(ytest_final2, 0, axis=1)

    print('x_train = ', x_train.shape)

    print('x_test = ', x_test.shape)

    print('xtest_final = ', xtest_final.shape)

    print('xtest_final2 = ', xtest_final2.shape)

    print('y_train = ', y_train.shape)

    print('y_test = ', y_test.shape)

    print('ytest_final = ', ytest_final.shape)

    print('ytest_final2 = ', ytest_final2.shape)

    return x_train, x_test, y_train, y_test, xtest_final, ytest_final, xtest_final2, ytest_final2


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if not title:
    #     if normalize:
    #         title = 'matrice de confusion normalisé'
    #     else:
    #         title = 'matrice de confusion sans normalisation'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("matrice de confusion normalisé")
    else:
        print('matrice de confusion sans normalisation')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='étiquette vrai',
           xlabel='étiquette prédite')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return cm, ax


def evaluation(loaded_model, x_testg, y_testg,title=None):
    np.set_printoptions(precision=2)

    score = loaded_model.evaluate(x_testg, y_testg)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    y_pred = loaded_model.predict(x_testg)

    y_testg = np.argmax(y_testg, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    # Plot non-normalized confusion matrix
    cm, ax = plot_confusion_matrix(y_testg, y_pred,
                                   classes=['Arrêter', 'Appeler', 'Saluer', 'Venir', 'Partir', 'Pointer'],
                                   title=title)

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_testg, y_pred, classes=['Arrêter', 'Appeler', 'Saluer', 'Venir', 'Partir', 'Pointer'],
                          normalize=True,
                          title=title)
    plt.show()

    precision = np.mean([cm[i][i] / np.sum(cm, axis=0)[i] for i in range(cm.shape[0])])
    print('Précision = ', precision)

    rappel = np.mean([cm[i][i] / np.sum(cm, axis=1)[i] for i in range(cm.shape[1])])
    print('Rappel = ', rappel)

    f_mesure = 2 * (precision * rappel) / (precision + rappel)
    print('F_mesure = ', f_mesure)

    tfp = np.mean([(np.sum(cm, axis=0)[i] - cm[i][i]) / (np.trace(cm) + np.sum(cm, axis=0)[i] - 2 * cm[i][i]) for i in
                   range(cm.shape[0])])
    print('TFP = ', tfp)

    tfn = np.mean([(np.sum(cm, axis=1)[i] - cm[i][i]) / np.sum(cm, axis=1)[i] for i in range(cm.shape[0])])
    print('TFN = ', tfn)

    return


if __name__ == "__main__":
    x = 12
    y = 18

    x_traing, x_testg, y_traing, y_testg, xTesting, yTesting, xTesting2, yTesting2 = load_data_skeleton()

    x_traing = x_traing.reshape(x_traing.shape[0], x, y, 1)

    x_testg = x_testg.reshape(x_testg.shape[0], x, y, 1)

    xTesting = xTesting.reshape(xTesting.shape[0], x, y, 1)

    xTesting2 = xTesting2.reshape(xTesting2.shape[0], x, y, 1)

    json_file = open('modelCNN986-0.06.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelCNN986-0.06.h5")
    print("Loaded Model from disk")

    # compile and evaluate loaded model
    loaded_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adagrad(),
                         metrics=['accuracy',
                                  metri.sparse_top_k_categorical_accuracy
                                  ])


    score = loaded_model.evaluate(x_traing, y_traing)

    print('Test loss:', score[0])
    print('Test accuracy training:', score[1])

    #evaluation(loaded_model, x_testg, y_testg)

    evaluation(loaded_model, xTesting, yTesting)

    #evaluation(loaded_model, xTesting2, yTesting2)

    loaded_model.summary()
