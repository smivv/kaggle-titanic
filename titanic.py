import os
import re
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


# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


def min_max_scale(data, cat_cols):
    # numeric attributes
    num_data = data.drop(cat_cols, axis=1)

    # scale to <0,1>
    num_data = pandas.DataFrame(MinMaxScaler().fit_transform(num_data))

    # fill nan with mean column values
    num_data.fillna(num_data.mean(), inplace=True)

    return num_data


def cat_vectorize(train_data, test_data, num_cols):
    # categorical attributes
    cat_train_data = train_data.drop(num_cols, axis=1)
    cat_test_data = test_data.drop(num_cols, axis=1)

    cat_train_data.fillna('NA', inplace=True)
    cat_test_data.fillna('NA', inplace=True)

    cat_train_data_values = cat_train_data.T.to_dict().values()
    cat_test_data_values = cat_test_data.T.to_dict().values()

    # vectorize (encode as one hot)
    vectorizer = DictVectorizer(sparse=False)
    vec_train_data = vectorizer.fit_transform(cat_train_data_values)
    vec_test_data = vectorizer.transform(cat_test_data_values)

    return vec_train_data, vec_test_data


DIR = os.path.dirname(os.path.realpath(__file__))

label_column = 'Survived'

cols = numpy.array(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
                    'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title'])

# numeric columns
num_cols = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# categorical columns
cat_cols = [item for item in cols if item not in (num_cols + [label_column])]

df_train = pandas.read_csv(os.path.join(DIR, 'data', 'train.csv'))
df_test = pandas.read_csv(os.path.join(DIR, 'data', 'test.csv'))

df_full = [df_train, df_test]

""" -------------------------------------- Feature Engineering ---------------------------------------- """

for dataset in df_full:
    dataset['Name_length'] = dataset['Name'].apply(len)

    # Feature that tells whether a passenger had a cabin on the Titanic
    dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # Create new feature FamilySize as a combination of SibSp and Parch
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Create new feature IsAlone from FamilySize
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    # Remove all NULLS in the Embarked column
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    # Remove all NULLS in the Fare column and create a new feature CategoricalFare
    dataset['Fare'] = dataset['Fare'].fillna(df_train['Fare'].median())

    # df_train['CategoricalFare'] = pandas.qcut(df_train['Fare'], 4)

    # Create a New feature CategoricalAge
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = numpy.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][numpy.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

    # df_train['CategoricalAge'] = pandas.cut(df_train['Age'], 5)

    # Create a new feature Title, containing the titles of passenger names
    dataset['Title'] = dataset['Name'].apply(get_title)

    # Group all non-common titles into one single grouping "Rare"
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

print(df_train.head(10))

# split the data into train and test
# x_train, _, y_train, _ = train_test_split(df_train.drop(label_column, axis=1),
#                                                     df_train[label_column],
#                                                     test_size=0)

x_train, y_train = df_train.drop(label_column, axis=1), df_train[label_column]
x_test = df_test

# x_train, y_train = df_train.loc[:, df_train.columns != label_column], df_train.loc[:, df_train.columns == label_column]
#
# x_test, y_test = df_test.loc[:, df_test.columns != label_column], df_test.loc[:, df_test.columns == label_column]


x_num_train = min_max_scale(x_train, cat_cols)
x_num_test = min_max_scale(x_test, cat_cols)

# labels or target attribute
y_train = y_train.astype(int)
# y_test = y_test.astype(int)

print(x_num_train[:10])

vec_x_cat_train, vec_x_cat_test = cat_vectorize(x_train, x_test, num_cols)

# build the feature vector
x_train = numpy.hstack((x_num_train, vec_x_cat_train))
x_test = numpy.hstack((x_num_test, vec_x_cat_test))

print(len(x_train), len(y_train))

# # build logistic regression model with class balancing
# lr = LogisticRegression(solver='lbfgs', class_weight='balanced')
# lr.fit(x_train, y_train.values)
# lr_y_pred = lr.predict(x_test)
#
# # print logistic regression result (they should be very poor)
# plot_classification_report(classification_report(y_test.values, lr_y_pred, digits=4))
# plot_confusion_matrix(y_test.values, lr_y_pred, labels=[0, 1])
# print("LR Accuracy:\n", accuracy_score(y_test.values, lr_y_pred), '\n')

# build logistic regression model with class balancing
# dt = DecisionTreeClassifier(class_weight='balanced')
# dt.fit(x_train, y_train.values)
# dt_y_pred = dt.predict(x_test)

# # print logistic regression result (they should be very poor)
# plot_classification_report(classification_report(y_test.values, dt_y_pred, digits=4))
# plot_confusion_matrix(y_test.values, dt_y_pred, labels=[0, 1])
# print("DTree Accuracy:\n", accuracy_score(y_test.values, dt_y_pred), '\n')
#
# # build logistic regression model with class balancing
# rf = RandomForestClassifier(class_weight='balanced')
# rf.fit(x_train, y_train.values)
# rf_y_pred = rf.predict(x_test)
#
# # print logistic regression result (they should be very poor)
# plot_classification_report(classification_report(y_test.values, rf_y_pred, digits=4))
# plot_confusion_matrix(y_test.values, rf_y_pred, labels=[0, 1])
# print("RForest Accuracy:\n", accuracy_score(y_test.values, rf_y_pred), '\n')


nb = GaussianNB()
nb.fit(x_train, y_train.values)
nb_y_pred = nb.predict(x_test)

pred = nb.predict(x_test)
submission = pandas.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': pred})
submission.to_csv(os.path.join(DIR, 'data', 'gender_submission.csv'), index=False)


