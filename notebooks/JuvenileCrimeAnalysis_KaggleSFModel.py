# loading all the needed packages
import time
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.native_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import Point, Polygon, shape
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Reading file from the filesystem
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")

sns.set(style = 'darkgrid')
sns.set_palette('PuBuGn_d')

df_total = pd.read_csv("../input/indiainsights-juvenilecrimes-1995-2005/IndiaGov_JuvenileCrimes_1995_2005.csv")
df_total.head(9)
df_total.iloc[:, 4].values

# Random Partitioning of the Dataset using Scikit Learn Train Test Splitter
X, Y = df_total.iloc[:, 0:].values, df_total.iloc[:, 1:].values, 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

# Measuring the length of the Training and Test Dataset
len (X_train), len(X_test)

# Attempting Data Normalisation using MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)
