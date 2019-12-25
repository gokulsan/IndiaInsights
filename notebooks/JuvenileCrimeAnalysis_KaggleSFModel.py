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

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")

sns.set(style = 'darkgrid')
sns.set_palette('PuBuGn_d')

train = pd.read_csv("../input/indiainsights-juvenilecrimes-1995-2005/IndiaGov_JuvenileCrimes_1995_2005.csv")
train.head(9)
train.shape
train.isnull().sum()

x = sns.catplot('Incidence of Juvenile Crimes', data = train, kind = 'count', aspect = 3, height = 4.5)
x.set_xticklabels(rotation = 85)

le = preprocessing.LabelEncoder()
jCrimes = le.fit_transform(train['Incidence of Juvenile Crimes'])

training, validation = train_test_split(jCrimes, train_size = 0.60)

start = time.time()
model = BernoulliNB()
model.fit(training[features], training['jCrimes'])
predicted = np.array(model.predict_proba(validation[features]))
end = time.time()
secs = (end - start)
loss = log_loss(validation['jCrimes'], predicted)
print("Total seconds: {} and loss {}".format(secs, loss))
