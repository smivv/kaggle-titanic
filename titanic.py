import os
import numpy
import pandas
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler

from utils import plot_classification_report, plot_confusion_matrix


import matplotlib.pyplot as plt

label_column = 'Survived'

cols = numpy.array(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
                    'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])

# numeric columns
num_cols = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# categorical columns
cat_cols = [item for item in cols if item not in (num_cols + [label_column])]

df_train = pandas.read_csv('/Users/smivv/PycharmProjects/kaggle-titanic/data/train.csv')
df_test = pandas.read_csv('/Users/smivv/PycharmProjects/kaggle-titanic/data/test.csv')

print(df_train.head(10))

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(df_train.drop(label_column, axis=1),
                                                    df_train[label_column],
                                                    test_size=0.2)

# X_train, y_train = df_train.loc[:, df_train.columns != label_column], df_train.loc[:, df_train.columns == label_column]
#
# X_test, y_test = df_test.loc[:, df_test.columns != label_column], df_test.loc[:, df_test.columns == label_column]

# numeric attributs
x_num_train = X_train.drop(cat_cols, axis=1)
x_num_test = X_test.drop(cat_cols, axis=1)

# scale to <0,1>
x_num_train = pandas.DataFrame(MinMaxScaler().fit_transform(x_num_train))
x_num_test = pandas.DataFrame(MinMaxScaler().fit_transform(x_num_test))

# fill nan with mean column values
x_num_train.fillna(x_num_train.mean(), inplace=True)
x_num_test.fillna(x_num_test.mean(), inplace=True)

# labels or target attribute
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print(x_num_train[:10])

# categorical attributes
cat_train = X_train.drop(num_cols, axis=1)
cat_test = X_test.drop(num_cols, axis=1)

cat_train.fillna('NA', inplace=True)
cat_test.fillna('NA', inplace=True)

x_cat_train = cat_train.T.to_dict().values()
x_cat_test = cat_test.T.to_dict().values()

# vectorize (encode as one hot)
vectorizer = DictVectorizer(sparse=False)
vec_x_cat_train = vectorizer.fit_transform(x_cat_train)
vec_x_cat_test = vectorizer.transform(x_cat_test)

# build the feature vector
x_train = numpy.hstack((x_num_train, vec_x_cat_train))
x_test = numpy.hstack((x_num_test, vec_x_cat_test))

print(len(x_train), len(y_train))

# build logistic regression model with class balancing
lr = LogisticRegression(solver='lbfgs', class_weight='balanced')
lr.fit(x_train, y_train.values)
lr_y_pred = lr.predict(x_test)

# print logistic regression result (they should be very poor)
plot_classification_report(classification_report(y_test.values, lr_y_pred, digits=4))
plot_confusion_matrix(y_test.values, lr_y_pred, labels=[0, 1])
print("Accuracy:\n", accuracy_score(y_test.values, lr_y_pred), '\n')
