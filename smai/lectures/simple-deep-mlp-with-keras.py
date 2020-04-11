from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy score
import pandas as pd
import numpy as np

# Read data
train = pd.read_csv('train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('test.csv').values).astype('float32')
labels_test = train.ix[:,1].values.astype('int32')

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels)

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical cross-entropy for the loss and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Training...")
model.fit(X_train, y_train, nb_epoch=10, batch_size=16, validation_split=0.1, show_accuracy=True, verbose=2)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)

print("Performance Measure Summary ..")
labels_pred = np.argmax(preds, axis=1)

# Print scores
print(accuracy_score(labels_test, labels_pred , average="macro")) # macro => all datapoints are treated equal
print(precision_score(labels_test, labels_pred , average="macro"))
print(recall_score(labels_test, labels_pred , average="macro"))
print(f1_score(labels_test, labels_pred , average="macro"))
