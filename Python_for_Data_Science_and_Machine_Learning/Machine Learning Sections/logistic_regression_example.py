import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ignore warnings of lapack bug
import warnings
warnings.filterwarnings(action="ignore", module="sklearn",
                        message="^internal gelsd")
warnings.simplefilter(action='ignore', category=FutureWarning)


# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

# import train and test data
train = pd.read_csv('tmp/train.csv')
test = pd.read_csv('tmp/test.csv')

print('*' * 40)
print(train.head(5))
print('*' * 40)
print(train.info())
print('*' * 40)
print(test.info())
print('*' * 40)

sns.set_style('whitegrid')

# setup multiple plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# look for missing data
sns.heatmap(train.isnull(), yticklabels=False,
            cbar=False, cmap='viridis', ax=ax[0])
sns.heatmap(test.isnull(), yticklabels=False,
            cbar=False, cmap='viridis', ax=ax[1])


# sns.countplot(data=train, x='Survived', hue='Sex',
#               palette='RdBu_r', ax=ax[1][0])
# sns.countplot(data=train, x='Survived', hue='Pclass', ax=ax[1][1])
# sns.distplot(train['Age'].dropna(), kde=False, bins=30, ax=ax[2][0])

# sibling - spouse
# sns.countplot(data=train, x='SibSp', ax=ax[2][0])

# sns.distplot(train['Fare'], kde=False, bins=40, ax=ax[2][1])

# sns.boxplot(data=train, x='Pclass', y='Age', ax=ax[2][1])
# sns.boxplot(data=test, x='Pclass', y='Age', ax=ax[3][0])


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


def impute_age2(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 42
        elif Pclass == 2:
            return 26
        else:
            return 23
    else:
        return Age


# clean up missing age with averages
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
test['Age'] = test[['Age', 'Pclass']].apply(impute_age2, axis=1)

# fare is has one NaN for test
test['Fare'].fillna(0, inplace=True)

# sns.heatmap(train.isnull(), yticklabels=False,
#             cbar=False, cmap='viridis', ax=ax[3][1])
# sns.heatmap(test.isnull(), yticklabels=False,
#             cbar=False, cmap='viridis', ax=ax[4][0])

# too much missing data for cabin, so drop it
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

# clean up data
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

# push pclass into new columns and drop the first column
pclass = pd.get_dummies(train['Pclass'], drop_first=True)
if 'pclass' in locals():
    train = pd.concat([train, sex, embark, pclass], axis=1)
    train.drop(['Sex', 'Embarked', 'Name', 'Ticket',
                'PassengerId', 'Pclass'], axis=1, inplace=True)
else:
    train = pd.concat([train, sex, embark], axis=1)
    train.drop(['Sex', 'Embarked', 'Name', 'Ticket',
                'PassengerId'], axis=1, inplace=True)

# clean up data
sex2 = pd.get_dummies(test['Sex'], drop_first=True)
embark2 = pd.get_dummies(test['Embarked'], drop_first=True)

# push pclass into new columns and drop the first column
pclass2 = pd.get_dummies(test['Pclass'], drop_first=True)
if 'pclass2' in locals():
    test = pd.concat([test, sex2, embark2, pclass2], axis=1)
    test.drop(['Sex', 'Embarked', 'Name', 'Ticket',
               'PassengerId', 'Pclass'], axis=1, inplace=True)
else:
    test = pd.concat([test, sex2, embark2], axis=1)
    test.drop(['Sex', 'Embarked', 'Name', 'Ticket',
               'PassengerId'], axis=1, inplace=True)

train.to_csv('tmp/processed_train.csv')
test.to_csv('tmp/processed_test.csv')

print('*' * 40)
print(train.info())
print('*' * 40)
print(test.info())
print('*' * 40)

# X = train.drop('Survived', axis=1)
# y = train['Survived']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=101)

X_train = train.drop('Survived', axis=1)
X_test = test
y_train = train['Survived']

logrm = LogisticRegression()
logrm.fit(X_train, y_train)

predictions = logrm.predict(X_test)
prediction = pd.DataFrame(predictions, columns=[
                          'predictions'])
prediction.reset_index(inplace=True)
# prediction = pd.concat([test.PassengerId, prediction], axis=1)
prediction.columns = ['PassengerId', 'Survived']
prediction['PassengerId'] += 892

print(prediction.head(5))
print(prediction.tail(5))
prediction.to_csv('tmp/prediction.csv', index=False)


# print(classification_report(y_test, predictions))
# print('*' * 40)
# print(confusion_matrix(y_test, predictions))
# print('*' * 40)

# tighten up layout
plt.tight_layout()

# show plot
plt.show()

# save plots
fig.savefig("machine_learning_logrm.png")
