# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np
import os
import tensorflow as tf

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
        element = "./cad60/%s" % (element)
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


# Function to create model, required for KerasClassifier
def create_model():
    input_shape=(12,12,1)
    outputshap=(18)
    # create model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1),
                                     activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(512, (4, 4), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(400, activation='relu'))
    model.add(tf.keras.layers.Dense(outputshap, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
# dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]

x_traing, x_testg, y_traing, y_testg = load_data_cad60()
# create model

model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]

param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_traing, y_traing)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
