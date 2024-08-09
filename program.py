#importhing libiraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#data loading
#enter the path of the train dataset below
titanic_data=pd.read_csv('train.csv')
#data preprocessing
titanic_data=titanic_data.drop(columns='Cabin', axis=1)#removing columns which are not useful
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)#replacing null values with their mean value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)#replacing mode value with missing values
#encoding categorical columns/data
titanic_data=titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}})
x=titanic_data.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
y=titanic_data['Survived']
#splitting the data into test data and train data
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=2)
#Logistical regression and model training
model=LogisticRegression()
model.fit(X_train,Y_train)
#predicting
X_train_predict=model.predict(X_train)
X_test_prdict=model.predict(X_test)
#testing the model
x_train_acc=accuracy_score(Y_train,X_train_predict)
print("train data accuracy value is:",x_train_acc)
x_test_acc=accuracy_score(Y_test,X_test_predict)
print("test data accuracy value is:",x_test_acc)