import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

dataset_path = 'D:/Python/data_analysis/Datasets/home-data-for-ml-course/train.csv'
dataset = pd.read_csv(dataset_path)
dataframe_dataset = pd.DataFrame(dataset)

print(dataset.describe())

filtered_dataset = dataframe_dataset.dropna(axis=0)

print(f'After data is filtered: ', filtered_dataset)

y = dataset.SalePrice

feature_cols = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = dataset[feature_cols]

melbourne_model = DecisionTreeRegressor(random_state=1)

train_X, validate_X, train_y, validate_y = train_test_split (X, y, train_size=0.8, test_size=0.2, random_state=1)

melbourne_model.fit(train_X,train_y)

melbourne_predictions = melbourne_model.predict(validate_X)

print(melbourne_predictions)

mae_melbourne_model = mean_absolute_error(validate_y, melbourne_predictions)

print(mae_melbourne_model)








