#D:/Python/data_analysis/Datasets/home-data-for-ml-course/

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Read data

train_file_path = 'D:/Python/data_analysis/Datasets/home-data-for-ml-course/train.csv'
test_file_path = 'D:/Python/data_analysis/Datasets/home-data-for-ml-course/test.csv'
train_dataset = pd.read_csv(train_file_path, index_col = 'Id')
test_dataset = pd.read_csv(test_file_path, index_col = 'Id')

#Remove rows with missing value in SalePrice column
train_dataset.dropna(axis = 0, subset = 'SalePrice', inplace = True)
y = train_dataset.SalePrice
train_dataset.drop(['SalePrice'], axis = 1, inplace = True)

X = train_dataset.select_dtypes(exclude = 'object')
test_X = test_dataset.select_dtypes(exclude = 'object')

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

print(X.head())

# Shape of training data (num_rows, num_columns)
print(train_X.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (train_X.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


def score_data(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(n_estimators = 100, random_state = 1)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    mae = mean_absolute_error(val_y, predictions)
    return mae


#Approach 1: Drop columns with missing values
cols_with_missing_value = [col for col in train_X.columns if train_X[col].isnull().any()]

filtered_train_X = train_X.drop(cols_with_missing_value, axis = 1)
filtered_val_X = val_X.drop(cols_with_missing_value, axis = 1)


print('MAE (Drop columns with missing values):')
print(score_data(filtered_train_X, filtered_val_X, train_y, val_y))

#Approach 2: Imputation

my_imputer = SimpleImputer()

imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_val_X = pd.DataFrame(my_imputer.transform(val_X))

imputed_train_X.columns = train_X.columns
imputed_val_X.columns = val_X.columns

print('MAE (Imputation)')
print(score_data(imputed_train_X, imputed_val_X, train_y, val_y))
