#Data folder path: D:\Python\data_analysis\Datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


dataset = pd.read_csv(r'D:/Python/data_analysis/datasets/Data.csv')
df = pd.DataFrame(dataset)
print(df)


X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

print(X)
print(y)


print(df.isnull().sum())

df1 = df.copy()

print('Before:', df1.shape)
df1.dropna(inplace=True)
print('After:', df1.shape)

df2= df.copy()

import warnings
warnings.filterwarnings('ignore')

# fill missing values with mean column values
df2.fillna(df2.mean(numeric_only=True), inplace=True)
# count the number of NaN values in each column
print(df2.isnull().sum())

print(df2)

print("Before:", X)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

print("After:", X)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(df)

print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, test_size=0.2, random_state=1)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
X_train[:,3:] = mm.fit_transform(X_train[:,3:])
X_test[:,3:] = mm.transform(X_test[:,3:])

print(X_train[:,3:])
print(X_test[:,3:])