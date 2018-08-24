import numpy as np
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5, 2)), range(5)

print(X)
print('*' * 40)
print(list(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print('*' * 40)
print(X_train)

print('*' * 40)
print(y_train)
print('*' * 40)
print(X_test)
print('*' * 40)
print(y_test)

# model.fit(X_train, y_train)
