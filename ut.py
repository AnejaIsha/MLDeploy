import numpy as np
import pandas as pd

import sklearn
boston = pd.read_csv('E:\\Anaconda\\Lib\\site-packages\\sklearn\\datasets\\data\\boston_house_prices.csv')
boston = boston[~(boston['MEDV'] >= 50.0)]
from sklearn import preprocessing
# Let's scale the columns before plotting them against MEDV
min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x = boston.loc[:,column_sels]
y = boston['MEDV']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
y =  np.log1p(y)
for col in x.columns:
    if np.abs(x[col].skew()) > 0.3:
        x[col] = np.log1p(x[col])
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import MinMaxScaler
model = LR()
kf = KFold(n_splits=10)
mm = MinMaxScaler()
x_scaled = mm.fit_transform(x)
from sklearn.pipeline import make_pipeline
from sklearn import linear_model as l
from sklearn.preprocessing import PolynomialFeatures
model =make_pipeline(PolynomialFeatures(degree = 3), l.Ridge())
model.fit(x_scaled,y)
import pickle
file_name = 'finalized_model.pickle'
pickle.dump(model,open(file_name,'wb'))