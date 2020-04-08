import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv(filepath_or_buffer="bank_info.csv", sep=";")

X = dataset.iloc[:,[0,1,2,3,4,5,6,7,11,12,13,14]]
y = dataset.iloc[:,-1:]

#Encode categroical data
ct = ColumnTransformer([('encoder',OneHotEncoder(),[1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

#Avoid dummy variable trap
X = X[:,1:]

ct = ColumnTransformer([('encoder',OneHotEncoder(),[12])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

#Avoid dummy variable trap
X = X[:,1:]

ct = ColumnTransformer([('encoder',OneHotEncoder(),[14])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

#Avoid dummy variable trap
X = X[:,1:]

ct = ColumnTransformer([('encoder',OneHotEncoder(),[17])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

#Avoid dummy variable trap
X = X[:,1:]

ct = ColumnTransformer([('encoder',OneHotEncoder(),[19])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

#Avoid dummy variable trap
X = X[:,1:]


ct = ColumnTransformer([('encoder',OneHotEncoder(),[20])], remainder="passthrough")
X = np.array(ct.fit_transform(X), dtype=np.float)

#Avoid dummy variable trap
X = X[:,1:]


ct = ColumnTransformer([('encoder',OneHotEncoder(),[0])], remainder="passthrough")
y = np.array(ct.fit_transform(y),dtype=np.int)
y = y[:,1:]

#feature Scaling

sc_X = StandardScaler()
X  = sc_X.fit_transform(X)

#Train and Test

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

#ANN Code

#Initializing the ANN: Defining sequence of layers
classifier = Sequential()

#Adding the input and first hidden layer
classifier.add(Dense(units = 14, kernel_initializer='uniform', activation='relu', input_dim=25))

#Adding second hidden layer
classifier.add(Dense(units = 14, kernel_initializer='uniform', activation='relu'))

#Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid'))

#Compiling ANN (SGD)
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#fitting the ANN to training set
classifier.fit(X_train,y_train,batch_size=10,epochs=10)

#Predict
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Confusion metrics
cm = confusion_matrix(y_test,y_pred)


#Accuracy
acc = (cm[0][0] + cm[1][1])/ np.sum(cm)
print("Accurancy is {0}".format(acc*100))

#Recall
recall = cm[1][1]/(cm[1][1] + cm[1][0])
print("Recall is {0}".format(recall*100))

#Precision
prec = cm[1][1]/(cm[1][1] + cm[0][1])
print("Precision is {0}".format(prec*100))

#F1 Score
f1_score = 2 * prec * recall/(prec+recall)
print("F1 Score {0}".format(f1_score))

















