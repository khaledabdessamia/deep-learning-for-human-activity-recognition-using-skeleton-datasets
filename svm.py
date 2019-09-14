import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pickle


train_data = np.empty(shape=(0, 218))
test_data = np.empty(shape=(0, 218))

hello = np.genfromtxt("./skeleton/Hello.csv", delimiter=',')
call = np.genfromtxt("./skeleton/Call.csv", delimiter=',')
stop = np.genfromtxt("./skeleton/Stop.csv", delimiter=',')
pointing = np.genfromtxt("./skeleton/Pointing.csv", delimiter=',')
coming = np.genfromtxt("./skeleton/Coming.csv", delimiter=',')
going = np.genfromtxt("./skeleton/Going.csv", delimiter=',')

training_pourcentage = 1

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


X = train_data[:,1:217]
print(X.shape)
y = train_data[:,0]
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear',C=0.03)
svclassifier.fit(X_train, y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(svclassifier, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

result = loaded_model.score(X_train, y_train)
print('result training =',result)

result = loaded_model.score(X_test, y_test)

print('result test =',result)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

true = 0
for i in range(len(y_test)):

    if y_test[i] == y_pred[1]:
        true += 1

true = true / len(y_test)
print(true)