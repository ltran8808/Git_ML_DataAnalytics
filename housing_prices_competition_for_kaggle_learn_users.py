import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

file_path = 'D:/Python/data_analysis/Datasets/home-data-for-ml-course/train.csv'
dataset = pd.read_csv(file_path)

# filtered_dataset = dataset.dropna(axis = 0)

y = dataset.SalePrice

feature_columns = ['LotArea', 'YearBuilt', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF','2ndFlrSF', 'TotRmsAbvGrd', 'GarageCars']

X = dataset[feature_columns]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

hpc_model = RandomForestRegressor(random_state = 1)

hpc_model.fit(train_X, train_y)

print(f'val_X array items are: {val_X}')

predictions = hpc_model.predict(val_X)

print(predictions)

mae = mean_absolute_error(val_y, predictions)

print(mae)
