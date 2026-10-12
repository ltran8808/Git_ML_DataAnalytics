#D:\Python\data_analysis\Datasets\melb_data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

file_path = 'D:/Python/data_analysis/Datasets/melb_data/melb_data.csv'
melb_data = pd.read_csv(file_path)

y = melb_data.Price

no_price_col_data = melb_data.drop(['Price'], axis=1)
X = no_price_col_data.select_dtypes(exclude=['object'])

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 1)

def score_data(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(n_estimators = 10, random_state = 1)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    mae = mean_absolute_error(val_y, predictions)
    return mae

# Get names of columns with missing values
missing_value_columns = [col for col in train_X.columns
                         if train_X[col].isnull().any()]

#Approach1: Drop columns in training and validation data
filtered_train_X = train_X.drop(missing_value_columns, axis = 1)
filtered_val_X = val_X.drop(missing_value_columns, axis = 1)

print('MAE from Approach 1 (Drop columns with missing values:)')
print(score_data(filtered_train_X, filtered_val_X, train_y, val_y))

#Approach2: SimpleImputer

my_imputer = SimpleImputer()
imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_val_X = pd.DataFrame(my_imputer.transform(val_X))

#Imputation removed columns. Add the columns back
imputed_train_X.columns = train_X.columns
imputed_val_X.columns = val_X.columns

print('MAE from Approach 2 (Imputation):')
print(score_data(imputed_train_X, imputed_val_X, train_y, val_y))


#Imputation with Extension (imputation with keeping track of imputed value)

#Make copy to avoid changing original data (when imputing)
copied_train_X = train_X.copy()
copied_val_X = val_X.copy()

#Create new columns and make each row with True or False to marked imputed value
for col in missing_value_columns:
    copied_train_X[col + "_was_missing"] = copied_train_X[col].isnull()
    copied_val_X[col + "_was_missing"] = copied_val_X[col].isnull()

#Imputation
my_imputer = SimpleImputer()
imputed_copied_train_X = pd.DataFrame(my_imputer.fit_transform(copied_train_X))
imputed_copied_val_X = pd.DataFrame(my_imputer.transform(copied_val_X))

#Add column name after imputation has removed them
imputed_copied_train_X.columns = copied_train_X.columns
imputed_copied_val_X.columns = copied_val_X.columns

print('MAE from Approach 3 (Imputation with Extension)')
print(score_data(imputed_copied_train_X, imputed_copied_val_X, train_y, val_y))

#shape of training data (num_rows, num_columns)
print(train_X.shape)

#number of missing values in each column of training data
missing_val_count_by_column = (train_X.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
