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

