#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Data loading
titanic_data = pd.read_csv('train.csv')

# Data Preprocessing
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data = titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}})

# Feature Selection
x = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
y = titanic_data['Survived']

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, Y_train)

# Random Forest Model (for feature importance)
rf_model = RandomForestClassifier(random_state=2)
rf_model.fit(X_train, Y_train)

# Predictions
logistic_predictions = logistic_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(Y_test, logistic_predictions))
print("Random Forest Accuracy:", accuracy_score(Y_test, rf_predictions))

# Feature Importance
feature_importance = pd.Series(rf_model.feature_importances_, index=x.columns).sort_values(ascending=False)
print("Feature Importance:\n", feature_importance)

# Visualization of Feature Importance
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance')
plt.show()

# Classification Report
print(classification_report(Y_test, rf_predictions))
