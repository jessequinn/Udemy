import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

df = pd.read_csv("./bank_note_data.csv")

# plt.figure(1)
sns.countplot(x='Class', data=df)

# plt.figure(2)
sns.pairplot(data=df, hue='Class')

scaler = StandardScaler()
scaler.fit(df.drop('Class', axis=1))

scaled_features = scaler.transform(df.drop('Class', axis=1))

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

y = df['Class']
X = df_feat

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

feat_cols = []

for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))

input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train, y=y_train, batch_size=20, num_epochs=5, shuffle=True)

classifier = tf.estimator.DNNClassifier(
    hidden_units=[10, 20, 10], n_classes=2, feature_columns=feat_cols)

classifier.train(input_fn=input_func, steps=500)

pred_fn = tf.estimator.inputs.pandas_input_fn(
    x=X_test, batch_size=len(X_test), shuffle=False)

# generator
predictions = list(classifier.predict(input_fn=pred_fn))

final_preds = []

for pred in predictions:
    final_preds.append(pred['class_ids'][0])

print(confusion_matrix(y_test, final_preds))

print(classification_report(y_test, final_preds))

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

rfc_preds = rfc.predict(X_test)

print(confusion_matrix(y_test, final_preds))

print(classification_report(y_test, final_preds))

# save plots
for i in range(1, plt.gcf().number + 1):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('tensorflow_figure_' + str(i) + '.png')
    plt.close()
