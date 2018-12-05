import tensorflow as tf
import numpy as np
import os

# 2018-12-02 17:49:01.393240: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
# 2018-12-02 17:49:01.394793: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 4. Tune using inter_op_parallelism_threads for best performance.
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
# OMP: Hint: This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

labels = np.genfromtxt("./activityLabel.csv", delimiter=',', dtype=None, encoding=None)

num = []
label = []
for e in labels:
    num.append(e[0])
    label.append(e[1])

x_train = np.empty(shape=(0, 172))
x_test = np.empty(shape=(0, 172))
y_train = np.empty(shape=(0, 18))
y_test = np.empty(shape=(0, 18))

for element in os.listdir('.'):
    if element.endswith('.txt'):
        data = np.genfromtxt(element, delimiter=',')

        x_train = np.concatenate((x_train, data[0:int(0.8 * len(data))]), axis=0)
        x_test = np.concatenate((x_test, data[int(0.8 * len(data)):]), axis=0)

        s = element.split(".")
        z = np.zeros(shape=18)
        z[num.index(int(s[0]))] = int(1)

        y_train = np.concatenate((y_train, [z for i in range(int(0.8 * len(data)))]), axis=0)
        y_test = np.concatenate((y_test, [z for i in range(len(data) - int(0.8 * len(data)))]), axis=0)

x_train = np.delete(x_train,x_train.shape[1]-1, 1)
x_test = np.delete(x_test,x_test.shape[1]-1, 1)

# mini batch gradient descent ftw
batch_size = 128
# 10 difference characters
num_classes = 18
# very short training time
epochs = 12


model = tf.keras.models.Sequential([
   # tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(18, activation=tf.nn.softmax)
])
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
print(x_train.shape)
print(x_test.shape)

print(y_train.shape)

print(y_test.shape)

model.fit(x_train, y_train
          ,batch_size=1
          ,epochs=epochs
          ,verbose=1
          ,validation_data=(x_test, y_test)
          )
score = model.evaluate(x_test, y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
