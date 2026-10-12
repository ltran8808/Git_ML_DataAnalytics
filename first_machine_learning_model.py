#Dataset folder: D:/Python/data_analysis/Datasets/home-data-for-ml-course/train.csv
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

iowa_file_path = 'D:/Python/data_analysis/Datasets/home-data-for-ml-course/train.csv'
iowa_data = pd.read_csv(iowa_file_path)
print(iowa_data.columns)
print(iowa_data.describe())

iowa_data = iowa_data.dropna(axis=0)

y = iowa_data.SalePrice

iowa_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = iowa_data[iowa_features]

iowa_model = DecisionTreeRegressor(random_state=1)
