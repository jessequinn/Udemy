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
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

# download nltk packages in this case we used 'stopwords'
# nltk.download_shell()

# messages = [line.rstrip() for line in open(
#     './Natural-Language-Processing/smsspamcollection/SMSSpamCollection')]
# print(len(messages))

# for mess_no, message in enumerate(messages[:10]):
#     print(mess_no, message + '\n')

messages = pd.read_csv('./Natural-Language-Processing/smsspamcollection/SMSSpamCollection',
                       sep='\t', names=['label', 'message'])

# print(messages.describe())
# print(messages.groupby('label').describe())

messages['length'] = messages['message'].apply(len)

messages['length'].plot.hist(bins=150)

# outliar
# print(messages[messages['length']==910]['message'].iloc[0])

messages.hist(column='length', by='label', bins=60, figsize=(12, 4))

# text normalization and preprocessing


def text_process(mess):
    '''
    1. remove punc
    2. remove stop words
    3. return list of clean text words
    '''

    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# print(messages['message'].head(5).apply(text_process))


# bag of words - can take a long time to process.
bow_transformer = CountVectorizer(
    analyzer=text_process).fit(messages['message'])

# word count test
# print(len(bow_transformer.vocabulary_))
mess4 = messages['message'][3]
# print(mess4)
bow4 = bow_transformer.transform([mess4])
# print(bow4)
# print(bow4.shape)
# print(bow_transformer.get_feature_names()[4068])

messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: {}'.format(messages_bow.shape))
print('Amount of Non Zero Occurrences: {}'.format(messages_bow.nnz))

sparsity = (100.0 * messages_bow.nnz /
            (messages_bow.shape[0] * messages_bow.shape[1]))
print('Sparsity: {}'.format(sparsity))

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf = tfidf_transformer.transform(messages_bow)

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

print(spam_detect_model.predict(tfidf4)[0])

all_pred = spam_detect_model.predict(messages_tfidf)

print(all_pred)

msg_train, msg_test, label_train, label_test = train_test_split(
    messages['message'], messages['label'], test_size=0.3)

# pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

print(classification_report(label_test, predictions))

# pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

print(classification_report(label_test, predictions))

# save plots
for i in range(1, plt.gcf().number + 1):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('nlp_figure_' + str(i) + '.png')
    plt.close()
