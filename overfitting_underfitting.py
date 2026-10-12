import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#For loop to try model with different values of max_leaf_nodes

def check_model_accuracy(max_leaf_nodes_value, train_X, validate_X, train_y, validate_y):
    
        model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes_value, random_state = 0)
        model.fit(train_X, train_y)
        predictions = model.predict(validate_X)
        mae = mean_absolute_error(validate_y, predictions)

        print(f'when max_leaf_nodes is: ', max_leaf_nodes_value)
        print(f'MAE value is: ', mae)
        return(mae)

#Load data
file_path = 'D:/Python/data_analysis/Datasets/home-data-for-ml-course/train.csv'
dataset = pd.read_csv(file_path)
y = dataset.SalePrice
features_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = dataset[features_columns]

#Training and validation datasets creation

train_X, validate_X, train_y, validate_y = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

#Model creation
model = DecisionTreeRegressor(max_leaf_nodes=500, random_state=1)

#Fit model
model.fit(train_X, train_y)

#Predict
predictions = model.predict(validate_X)

print(f'Predictions are: ', predictions)

mae = mean_absolute_error(validate_y, predictions)

print(f'Mean Absolute Error value is: ', mae)



#Use Dictionary Comprehesion to store the best max_leaf_nodes value
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

scores={leaf_size: check_model_accuracy(leaf_size, train_X, validate_X, train_y, validate_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key = scores.get)

print(f'Best Tree Size is: {best_tree_size}')