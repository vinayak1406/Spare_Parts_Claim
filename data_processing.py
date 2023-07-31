import pandas as pd
from sklearn.model_selection import train_test_split

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
from sklearn.preprocessing import LabelEncoder










def preprocess_data(data1):
    # Preprocessing dats
    dealers = data1.get('Dealers')
 
    dealers['Dealer_ID'] = dealers['Dealer_ID'].fillna(0).astype(int)



    dealers['ZIpCode'] = dealers['ZIpCode'].fillna(0).astype(int)



    dealers.rename(columns = {'ZIpCode':'zipcode'}, inplace = True)


    customers = data1.get('Customers')
    customers['Cust_ID'] = customers['Cust_ID'].fillna(0).astype(int)


    customers['Dealer_ID'] = customers['Dealer_ID'].fillna(0).astype(int)

    customers['ZipCode'] = customers['ZipCode'].fillna(0).astype(int)

    customers.rename(columns = {'ZipCode':'zipcode'}, inplace = True)
    
   
    claims = data1.get('Claims')
    claims['Dealer_ID'] = claims['Dealer_ID'].fillna(0).astype(int)


    claims['claim_id'] = claims['claim_id'].fillna(0).astype(int)



    claims['claim_amount'] = claims['claim_amount'].fillna(0).astype(int)




    claims['Cust_ID'] = claims['Cust_ID'].fillna(0).astype(int)



    claims['Part_ID'] = claims['Part_ID'].fillna(0).astype(int)
    
    parts = data1.get('Parts')
   
    parts['Part_ID'] = parts['Part_ID'].fillna(0).astype(int)



    parts['Manufacturer_ID'] = parts['Manufacturer_ID'].fillna(0).astype(int)

    
    transactions = data1.get('Transactions')
    transactions['transaction_id'] = transactions['transaction_id'].fillna(0).astype(int)



    transactions['transaction_amount'] = transactions['transaction_amount'].fillna(0).astype(int)

    transactions['claim_id'] = transactions['claim_id'].fillna(0).astype(int)
    vendors = data1.get('Vendors')
    vendors['Vendor_ID'] = vendors['Vendor_ID'].fillna(0).astype(int)



    vendors['ZIpCode'] = vendors['ZIpCode'].fillna(0).astype(int)



    vendors.rename(columns = {'ZIpCode':'zipcode'}, inplace = True)
    
    return dealers, customers, claims, parts, transactions, vendors

def encode_labels(df):
    labelencoder = LabelEncoder()
    return df
