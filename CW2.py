

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.tree import export_text
import warnings
import os
    
def print_tree_structure(model, header_list):
    tree_rules = export_text(model, feature_names=header_list[:-1])
    print(tree_rules)
    
def load_data(file_path, delimiter=',') -> tuple:
    """
    Load the data from the CSV file and give back the number of rows

    :param file_path: path to the CSV file
    :param delimiter: delimiter used in the CSV file
    :return num_rows: number of rows in the CSV file
    :return data: data in the CSV file
    :return header_list: list of headers in the CSV file
    """
    
    num_rows, data, header_list=None, None, None

    if not os.path.isfile(file_path):
        warnings.warn(f"Task 1: Warning - CSV file '{file_path}' does not exist.")
        return None, None, None
    
    data = pd.read_csv(file_path, sep=delimiter) # read data from the file and set the delimiter variable
    num_rows = data.shape[0]
    header_list = data.columns.tolist()
    data = data.to_numpy()
    return num_rows, data, header_list

def filter_data(data) -> np.ndarray:
    """
    Give back the data by removing the rows with missing values

    :param data: data to be filtered
    :return filtered_data: data containing the rows without missing values
    """

    ifmissing = np.any(data == -99, axis=1) # checks for -99 in each row
    filtered_data = data[np.logical_not(ifmissing)] # removes rows with missing data

    return filtered_data

def statistics_data(data):
    """
    Calculate the coefficient of variation

    :param data: data to calculate the coefficient of variation
    :return coefficient_of_variation: coefficient of variation
    """

    coefficient_of_variation=None
    data=filter_data(data)

    isnan = np.any(np.isnan(data), axis=1)# checks for nan in each row
    data = data[np.logical_not(isnan)] # removes rows with nan data

    coefficient_of_variation = np.std(data, axis=0)  / np.mean(data, axis=0)
    print(coefficient_of_variation)

    return coefficient_of_variation

def split_data(data, test_size=0.3, random_state=1):
    """
    Split the dataset into training and testing 
    
    :param data: data to be split
    :param test_size: size of the testing set
    :param random_state: random state
    """
    x_train, x_test, y_train, y_test = None, None, None, None
    np.random.seed(1)

    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=test_size, train_size=(1-test_size), random_state=1, stratify=data[:, -1])

    return x_train, x_test, y_train, y_test

def train_decision_tree(x_train, y_train, ccp_alpha=0):
    """
    Train a decision tree model with the cost complexity parameter 

    :param x_train: training features
    :param y_train: training labels
    :param ccp_alpha: cost complexity parameter
    :return model: decision tree model
    """

    model=None

    model = DecisionTreeClassifier(ccp_alpha=ccp_alpha).fit(X=x_train, y=np.round(y_train))

    return model

def make_predictions(model, X_test):
    """
    Make predictions on the testing set
    
    :param model: trained model
    :param X_test: testing features
    :return y_test_predicted: predicted label
    """
    y_test_predicted=None

    y_test_predicted = model.predict(X_test)

    return y_test_predicted

def evaluate_model(model, x, y):
    """
    Evaluate the model performance by taking test dataset and giving back the accuracy and recall

    :param model: trained model
    :param x: test features
    :param y: test labels
    :return accuracy: accuracy of the model
    :return recall: recall of the model
    """
    accuracy, recall=None,None

    y_test_predicted = make_predictions(model, x)
    accuracy = accuracy_score(y, y_test_predicted)
    recall = recall_score(y, y_test_predicted)

    return accuracy, recall

def optimal_ccp_alpha(x_train, y_train, x_test, y_test):
    """
    Gives the optimal value for cost complexity parameter

    :param x_train: training features
    :param y_train: training labels
    :param x_test: testing features
    :param y_test: testing labels
    :return optimal_ccp_alpha: optimal value for cost complexity parameter
    """
    optimal_ccp_alpha = 0
    unpruned_model = train_decision_tree(x_train, y_train) # call task 5
    unpruned_accuracy, _ = evaluate_model(unpruned_model, x_test, y_test) # call task 7

    ccp_alpha = 0.001
    pruned_model = train_decision_tree(x_train, y_train, ccp_alpha) # call task 5
    pruned_accuracy, _ = evaluate_model(pruned_model, x_test, y_test) # call task 7

    while not (pruned_accuracy < unpruned_accuracy - 0.01 or pruned_accuracy > unpruned_accuracy + 0.01): # continually loop until found an accuract similar to unpruned model
        optimal_ccp_alpha = ccp_alpha
        ccp_alpha += 0.001
        pruned_model = train_decision_tree(x_train, y_train, ccp_alpha) # call task 5
        pruned_accuracy, _ = evaluate_model(pruned_model, x_test, y_test) # call task 7

    return optimal_ccp_alpha

def tree_depths(model):
    """
    Gives the depth of a decision tree that it takes as input.

    :param model: decision tree model
    :return depth: the depth of the model
    """
    depth=None

    depth = model.get_depth()
    
    return depth

def important_feature(x_train, y_train, header_list):
    """
    Trains a decision tree model and increases Cost Complexity Parameter until the depth reaches 1

    :param x_train: training features
    :param y_train: training labels
    :param header_list: list of headers in the CSV file
    :return best_feature: the remaining feature
    """
    best_feature=None

    ccp_alpha = 0.000
    model = train_decision_tree(x_train, y_train, ccp_alpha) # call task 5

    while not(tree_depths(model) == 1): # call task 9
        ccp_alpha += 0.001
        model = train_decision_tree(x_train, y_train, ccp_alpha)

    best_feature = str(header_list[np.argmax(model.feature_importances_)])

    return best_feature

if __name__ == "__main__":
    # Load data
    file_path = "DT.csv"
    num_rows, data, header_list = load_data(file_path)
    print(f"Data is read. Number of Rows: {num_rows}"); 
    print("-" * 50)

    # Filter data
    data_filtered = filter_data(data)
    num_rows_filtered=data_filtered.shape[0]
    print(f"Data is filtered. Number of Rows: {num_rows_filtered}"); 
    print("-" * 50)

    # Data Statistics
    coefficient_of_variation = statistics_data(data_filtered)
    print("Coefficient of Variation for each feature:")
    for header, coef_var in zip(header_list[:-1], coefficient_of_variation):
        print(f"{header}: {coef_var}")
    print("-" * 50)

    # Split data
    x_train, x_test, y_train, y_test = split_data(data_filtered)
    print(f"Train set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print("-" * 50)
    
    # Train initial Decision Tree
    model = train_decision_tree(x_train, y_train)
    print("Initial Decision Tree Structure:")
    print_tree_structure(model, header_list)
    print("-" * 50)
    
    # Evaluate initial model
    acc_test, recall_test = evaluate_model(model, x_test, y_test)
    print(f"Initial Decision Tree - Test Accuracy: {acc_test:.2%}, Recall: {recall_test:.2%}")
    print("-" * 50)

    # Train Pruned Decision Tree
    model_pruned = train_decision_tree(x_train, y_train, ccp_alpha=0.002)
    print("Pruned Decision Tree Structure:")
    print_tree_structure(model_pruned, header_list)
    print("-" * 50)

    # Evaluate pruned model
    acc_test_pruned, recall_test_pruned = evaluate_model(model_pruned, x_test, y_test)
    print(f"Pruned Decision Tree - Test Accuracy: {acc_test_pruned:.2%}, Recall: {recall_test_pruned:.2%}")
    print("-" * 50)

    # Find optimal ccp_alpha
    optimal_alpha = optimal_ccp_alpha(x_train, y_train, x_test, y_test)
    print(f"Optimal ccp_alpha for pruning: {optimal_alpha:.4f}")
    print("-" * 50)

    # Train Pruned and Optimized Decision Tree
    model_optimized = train_decision_tree(x_train, y_train, ccp_alpha=optimal_alpha)
    print("Optimized Decision Tree Structure:")
    print_tree_structure(model_optimized, header_list)
    print("-" * 50)
    
    # Get tree depths
    depth_initial = tree_depths(model)
    depth_pruned = tree_depths(model_pruned)
    depth_optimized = tree_depths(model_optimized)
    print(f"Initial Decision Tree Depth: {depth_initial}")
    print(f"Pruned Decision Tree Depth: {depth_pruned}")
    print(f"Optimized Decision Tree Depth: {depth_optimized}")
    print("-" * 50)
    
    # Feature importance
    important_feature_name = important_feature(x_train, y_train,header_list)
    print(f"Important Feature for Fraudulent Transaction Prediction: {important_feature_name}")
    print("-" * 50)
        
# References: 
# Line 53 is inspired by code at https://www.geeksforgeeks.org/how-to-check-whether-specified-values-are-present-in-numpy-array/
# Line 90 is inspired by code at https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# Line 107 is inspired by code at https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.fit
# Line 205 was inspired by code at https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html