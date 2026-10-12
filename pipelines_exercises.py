#Dataset folder: D:\Python\data_analysis\Datasets\home-data-for-ml-course
import pandas as pd
from sklearn.model_selection import train_test_split

#Read the data
X_full = pd.read_csv('D:/Python/data_analysis/Datasets/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('D:/Python/data_analysis/Datasets/home-data-for-ml-course/test.csv', index_col='Id')

#Remote rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

#Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

#"Cardinality" means the number of unique values in a column
#Select categorical columns with relatively low cardinality (convinient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if 
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

#Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if
                  X_train_full[cname].dtype in ['int64', 'float64']]

#Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()