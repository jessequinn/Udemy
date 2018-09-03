import nltk
from nltk.corpus import stopwords

import pandas as pd
import string

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

yelp = pd.read_csv('./Natural-Language-Processing/yelp.csv')

print(yelp.head(2))
print(yelp.info())
print(yelp.describe())

# a new column based on text length
yelp['text length'] = yelp['text'].apply(len)

sns.set_style('white')

g = sns.FacetGrid(yelp, col='stars')
g.map(plt.hist, 'text length', bins=50)

plt.figure(2)
sns.boxplot(x='stars', y='text length', data=yelp)

plt.figure(3)
sns.countplot(x='stars', data=yelp, palette='rainbow')

stars = yelp.groupby('stars').mean()
print(stars)

print(stars.corr())

plt.figure(4)

sns.heatmap(stars.corr(), cmap='coolwarm', annot=True)

yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]

X = yelp_class['text']
y = yelp_class['stars']

cv = CountVectorizer()

X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

nb = MultinomialNB()
nb.fit(X_train, y_train)

predictions = nb.predict(X_test)

print('-'*40)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print('-'*40)

# pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

X = yelp_class['text']
y = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print('-'*40)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print('-'*40)

# pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('classifier', RandomForestClassifier())
])

X = yelp_class['text']
y = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print('-'*40)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print('-'*40)

# save plots
for i in range(1, plt.gcf().number + 1):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('nlp_figure_' + str(i) + '.png')
    plt.close()
