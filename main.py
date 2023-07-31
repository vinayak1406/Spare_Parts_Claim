import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import seaborn as sns
import math
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_processing import preprocess_data, encode_labels
from modeling import (usecase_1,predict_fail_parts_by_location,
                      predict_fail_parts_by_mileage, predict_fail_parts_by_age,
                      generate_feedback_ratings, train_decision_tree_classifier)

# Read the data
data1 = pd.read_excel('D:/DS_Project/Dataset.xlsx', sheet_name=['Dealers', 'Customers', 'Claims', 'Parts', 'Transactions', 'Vendors'])

# Data preprocessing
dealers, customers, claims, parts, transactions, vendors = preprocess_data(data1)

# Encoding labels
dealers = encode_labels(dealers)
customers = encode_labels(customers)
claims = encode_labels(claims)
parts = encode_labels(parts)
transactions = encode_labels(transactions)
vendors = encode_labels(vendors)

# Perform Use Case 1 analysis
usecase_1(claims)

print("__________________________________________________________")


# Perform Use Case 2 analysis
# usecase_2(claims, parts)

print("__________________________________________________________")


# Predict parts that may fail according to location, mileage, and age
predict_fail_parts_by_location(claims, dealers)

print("__________________________________________________________")


predict_fail_parts_by_mileage(claims)

print("__________________________________________________________")

predict_fail_parts_by_age(claims, parts)

print("__________________________________________________________")


# Generate random feedback and ratings
generate_feedback_ratings(claims)

# Train and evaluate models
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# dtc_model = train_decision_tree_classifier(X_train, y_train)
# rfc_model = train_random_forest_classifier(X_train, y_train)
# rfr_model = train_random_forest_regressor(X_train, y_train)

# modeling.rfc.predict([[5,902]])
# print(train_decision_tree_classifier.rfc.predict([[5,902]]))
# result = a.train_decision_tree_classifier()
# print(result)
import warnings 
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names")
rfc_model = train_decision_tree_classifier(claims)
prediction = rfc_model.predict([[5, 902]])
print("RFC Prediction:", prediction)