import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# Just disables the warning, doesn't enable AVX/FMA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
hello = tf.constant('Hello World')

# print(type(hello))

sess = tf.Session()

print(sess.run(hello))

# constants
x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition: ', sess.run(x+y))
    print('Subtraction: ', sess.run(x-y))
    print('Multiplication: ', sess.run(x*y))
    print('Division: ', sess.run(x/y))

# placeholders (lacks constants or values)
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

add = tf.add(x, y)
sub = tf.subtract(x, y)
mul = tf.multiply(x, y)
div = tf.divide(x, y)

with tf.Session() as sess:
    print('Operations with Placeholders')
    print('Addition: ', sess.run(add, feed_dict={x: 20, y: 30}))
    print('Subtraction: ', sess.run(sub, feed_dict={x: 20, y: 30}))
    print('Multiplication: ', sess.run(mul, feed_dict={x: 20, y: 30}))
    print('Division: ', sess.run(div, feed_dict={x: 20, y: 30}))

a = np.array([[5.0, 5.0]])
b = np.array([[2.0], [2.0]])

mat1 = tf.constant(a)
mat2 = tf.constant(b)

matrix_multi = tf.matmul(mat1, mat2)

with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)

######################################################
############ MNIST Multi-Layer Perception ############
######################################################

mnist = input_data.read_data_sets('./tmp', one_hot=True)

# print(type(mnist))

# print(mnist.train.images[2].reshape(28, 28))

# sample = mnist.train.images[2034].reshape(28, 28)
sample = mnist.train.images[1].reshape(28, 28)

plt.figure(1)
plt.imshow(sample, cmap='gist_gray')

sample = mnist.train.images[1].reshape(784, 1)

plt.figure(2)
plt.imshow(sample, cmap='gist_gray', aspect=0.02)

# x value
x = tf.placeholder(tf.float32, shape=[None, 784])
# ten possible numbers for the 28x28 pixel
# weight
W = tf.Variable(tf.zeros([784, 10]))
# bias
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b

y_true = tf.placeholder(tf.float32, shape=[None, 10])

# defining error
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)

        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    matches = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))

    acc = tf.reduce_mean(tf.cast(matches, tf.float32))

    print(sess.run(acc, feed_dict={
          x: mnist.test.images, y_true: mnist.test.labels}))

######################################################
######################## IRIS ########################
######################################################

df = pd.read_csv("./iris.csv")

df.columns = ['sepal_length', 'sepal_width',
              'petal_length', 'petal_width', 'target']

df['target'] = df['target'].apply(int)

y = df['target']
X = df.drop('target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

feat_cols = []

for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))

input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train, y=y_train, batch_size=10, num_epochs=5, shuffle=True)

classifier = tf.estimator.DNNClassifier(
    hidden_units=[10, 20, 10], n_classes=3, feature_columns=feat_cols)

classifier.train(input_fn=input_func, steps=50)

pred_fn = tf.estimator.inputs.pandas_input_fn(
    x=X_test, batch_size=len(X_test), shuffle=False)

# generator
predictions = list(classifier.predict(input_fn=pred_fn))

final_preds = []

for pred in predictions:
    final_preds.append(pred['class_ids'][0])

print(confusion_matrix(y_test, final_preds))

print(classification_report(y_test, final_preds))

# save plots
for i in range(1, plt.gcf().number + 1):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('tensorflow_figure_' + str(i) + '.png')
    plt.close()
