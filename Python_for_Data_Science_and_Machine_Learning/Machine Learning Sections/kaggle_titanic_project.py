# load packages
import sys  # access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd  # collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

# collection of functions for scientific and publication-ready visualization
import matplotlib
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np  # foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

# collection of functions for scientific computing and advance mathematics
import scipy as sp
print("SciPy version: {}". format(sp.__version__))

import IPython
from IPython import display  # pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__))

import sklearn  # collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

# misc libraries
import random
import time

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

# Input data files are available in the "tmp/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "tmp"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

# Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

# Configure Visualization Defaults
# %matplotlib inline = show plots in Jupyter Notebook browser
# %matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12, 8

# import data from file: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
data_raw = pd.read_csv('tmp/train.csv')

# a dataset should be broken into 3 splits: train, test, and (final) validation
# the test file provided is the validation file for competition submission
# we will split the train set into train and test data in future sections
data_val = pd.read_csv('tmp/test.csv')

# to play with our data we'll create a copy
# remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs
data1 = data_raw.copy(deep=True)

# however passing by reference is convenient, because we can clean both datasets at once
data_cleaner = [data1, data_val]


# preview data
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
print(data_raw.info())
# data_raw.head() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html
# data_raw.tail() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html
data_raw.sample(10)
