##The Credit Card Fraud Detection project is used to identify whether a new transaction is fraudulent or not by modeling past credit card transactions with the knowledge of the ones that turned out to be fraud. We will use various predictive models to see how accurate they are in detecting whether a transaction is a normal payment or a fraud. Classification techniques are the promising solutions to detect the fraud and non-fraud transactions. Unfortunately, in a certain condition, classification techniques do not perform well when it comes to huge numbers of differences in data distribution. 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import stats

data = pd.read_csv("creditcard.csv")

print(data)
print(data.shape)

#searching if there is any null value
data.isnull().values.any()

#initialising fraud and non-fraud values

fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
print(fraud.shape, valid.shape)

##Most of the transactions are non-fraud. If we use this dataframe as the base for our predictive models and analysis we might get a lot of errors and our algorithms will probably overfit since it will "assume" that most transactions are not fraud. But we don't want our model to assume, we want our model to detect patterns that give signs of fraud! The data set is highly skewed, consisting of 492 frauds in a total of 284,807 observations. This resulted in only 0.172% fraud cases. This skewed set is justified by the low number of fraudulent transactions.Now that we have the data, we are using only 3 parameters for now in training the model (Time, Amount, and Class). 

#dividing X and Y from the datset

X = data[['Time', 'Amount']]
Y = data['Class']

print(X.shape)
print(Y.shape)

xData = X.values
yData = Y.values

#using scikit learn to split train and test dataset 
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 40)

##Using Random forest classifier, The random forest is a supervised learning algorithm that randomly creates and merges multiple decision trees into one “forest.” The goal is not to rely on a single learning model, but rather a collection of decision models to improve accuracy  

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc = rfc.fit(xTrain, yTrain)

#prediction
y_pred = rfc.predict(xTest)

#finding accuracy
from sklearn.metrics  import accuracy_score
acc = accuracy_score(yTest,y_pred)
print("the accuray for render forest is {}".format(acc))

##Using naive bayes classifier

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(xTrain, yTrain)
y_pred = nb.predict(xTest)
acc = accuracy_score(yTest, y_pred)
print("The accuracy for render forest is {}".format(acc))

##Using Dummy Classifier 

from sklearn.dummy import DummyClassifier
dc = DummyClassifier()
dc.fit(xTrain, yTrain)
y_pred = dc.predict(xTest)
acc = accuracy_score(yTest, y_pred)
print("The accuraccy of render forest is {}".format(acc))

##Using SVM classifier

from sklearn.svm import SVC
svm = SVC()
svm.fit(xTrain, yTrain)
y_pred = svm.predict(xTest)
acc = accuracy_score(yTest, y_pred)
print("The accuraccy of render forest is {}".format(acc))

print(fraud.shape)

##To create our balanced training data set, We calculated all of the fraudulent transactions in our data set . Then, We randomly selected the same number of non-fraudulent transactions and concatenated the two. There are 492 cases of fraud in our dataset so we can randomly get 492 cases of non-fraud to create our new sub dataframe. 


# Lets shuffle the data before creating the subsamples
data1 = data.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_data1 =  data1.loc[data1['Class'] == 1 ]
non_fraud_data1 = data1.loc[data1['Class'] == 0][:492]

#lets concatenate both data
normal_distributed_data1 = pd.concat([fraud_data1, non_fraud_data1])

#lets shuffle the dataframe
new_data1 = normal_distributed_data1.sample(frac = 1, random_state = 42)

new_data1.head()

##Down-Sizing is down-sizing method, closely related to the over-sampling method, that was considered in this category (rand_downsize) consists of eliminating, at random, elements of the over-sized class until it matches the size of the other class. 

print("Description of classes in subsample of datsset")
print(new_data1['Class'].value_counts()/len(new_data1))

import seaborn as sns
      
sns.countplot('Class',data = new_data1)
plt.title("Equally distriuted classes")
plt.show()

#lets divide dataset into X and Y
X = new_data1[['Time', 'Amount']]
Y = new_data1['Class']
print(X.shape)
print(Y.shape)
xData = X.values
yData = Y.values

##Finally appllying classification

from sklearn.model_selection import train_test_split
xTrain,xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 40)

#applying random forest
from sklearn.ensemble import RandomForestClassifier
rdc = RandomForestClassifier()
rdc.fit(xTrain,yTrain)

#prediction
y_pred = rdc.predict(xTest)

#finding accurcy
acc = accuracy_score(yTest, y_pred)
print("The accuracy of render forest is  {}".format(acc))