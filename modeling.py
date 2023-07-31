import random

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
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def usecase_1(claims):

# #  USECASE - 1

# 
# # top dealers that are processing the extended warranty Of parts 



#TOTAL NO OF DEALERS ARE


    claims.Dealer_ID.unique().shape                 #Total no of dealers are
    print("Total no of dealers are")
    print(claims.Dealer_ID.value_counts().head(10))
    claims['Dealer_ID'].value_counts



    print("___________________________________________________________________")

    claims['Dealer_ID'].nlargest(n=20) 
    print("Top 20 Dealers Which Are Processing claims")  
    print(claims)    #Top 20 Dealers Which Are Processing claims



    sns.set(rc={'figure.figsize':(29,23)})
    plt.xticks(rotation=90)
    sns.countplot(x =claims.Dealer_ID, data = claims,hue = claims.Repair_or_Replace )
    print("___________________________________________________________________")



    print("no of claims with repair of parts")
    print(claims['Repair_or_Replace'].value_counts()['rpr']) #no of claims with repair of parts


    print("no of claims with replacemet of parts")
    print(claims['Repair_or_Replace'].value_counts()['rplc']) #no of claims with replacemet of parts




    sns.set(rc={'figure.figsize':(29,23)})
    plt.xticks(rotation=90)
    sns.countplot(x =claims.Dealer_ID, data = claims,hue = claims.Repair_or_Replace )

    print("___________________________________________________________________")




    claims['Part_ID'].unique()     #no of parts in the data



    plt.xticks(rotation=90)
    sns.countplot(x =claims.Part_ID, data = claims,hue = claims.Repair_or_Replace )


    claims['Part_ID'].value_counts().nlargest(n=7)

    counts = claims.groupby('Part_ID')['Repair_or_Replace'].value_counts().reset_index(name='counts')

    print(counts)



    sns.set_style('whitegrid')
    sns.barplot(x='Part_ID', y='counts', hue='Repair_or_Replace', data=counts)
    plt.title('Number of Repairs and Replacements for Each Part')
    plt.xlabel('Part ID')
    plt.ylabel('Count')
    plt.show()
    print("___________________________________________________________________")
    print("___________________________________________________________________")

    #___________________
    # ____________________________________________________________________________________________________________________
    #_________________________________________________________________________________________________________________________________________

    # def usecase_2(claims, parts):
    #     # ... perform analysis for Use Case 2
    #     # ...

def predict_fail_parts_by_location(claims, dealers):
    location['claim_date'] = pd.to_datetime(location['claim_date'])




    location['claim_date'] = location['claim_date'].dt.strftime('%Y%m%d')




    location


    df = location.copy()




    df




    labelencoder = LabelEncoder()
    df['claim_date'] = labelencoder.fit_transform(df['claim_date'])
    df['Dealer_ID'] = labelencoder.fit_transform(df['Dealer_ID'])
    df['Repair_or_Replace'] = labelencoder.fit_transform(df['Repair_or_Replace'])
    df['State'] = labelencoder.fit_transform(df['State'])
    df['City'] = labelencoder.fit_transform(df['City'])
    df['zipcode'] = labelencoder.fit_transform(df['zipcode'])
    df['Dealer_Name'] = labelencoder.fit_transform(df['Dealer_Name'])
    df['Dealer_Country'] = labelencoder.fit_transform(df['Dealer_Country'])


    # #  Train the decision tree classifier model for parts which may fails according to location.




    X = df[['Repair_or_Replace','Dealer_ID','Dealer_Name','Dealer_Country', 'State', 'City', 'zipcode']]
    y = df['Part_ID']




    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)



    clf = LogisticRegression(random_state=1)
    clf.fit(X_train, y_train)




    y_pred = clf.predict(X_test)




    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)





    df.corr()

def predict_fail_parts_by_mileage(claims):
    print("___________________________________________________________________________________________________")
    # #  Predictions for parts which may fails according to mileage which is total kms completed by part_id Using Random forest regressor and  linear regression




    claims['mileage'] = [random.randint(5000, 100000) for _ in range(len(claims))]




    claims





    claims.corr()




    data = claims.copy()



    # data.to_csv("D:/gsf/data.csv")




    labelencoder = LabelEncoder()
    data['mileage'] = labelencoder.fit_transform(data['mileage'])
    data['Part_ID'] = labelencoder.fit_transform(data['Part_ID'])
    data['Repair_or_Replace'] = labelencoder.fit_transform(data['Repair_or_Replace'])
    data['claim_date'] = labelencoder.fit_transform(data['claim_date'])




    data.describe()




    X = data['mileage'].values.reshape(-1,1)
    y = data['Part_ID'].values.reshape(-1,1)




    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)



    y_pred = rf.predict(X_test)




    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-Squared Score:", r2)
    accuracy = rf.score(X, y)
    print('Model accuracy (R^2):', accuracy)


    rf.predict([[6000]])



    print("__________________________________________________________")

def predict_fail_parts_by_age(claims, parts):
    # #  Prediction for parts which may fails according to age




    age = pd.merge(claims, parts, on="Part_ID")




    # Convert the claim_date and Manf_Date columns to datetime format
    age["claim_date"] = pd.to_datetime(age["claim_date"])
    age["Manf_Date"] = pd.to_datetime(age["Manf_Date"])



    age["age"] = (age["claim_date"] - age["Manf_Date"]).dt.days








    y =  age['age'].values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    arf = RandomForestRegressor(n_estimators=100, random_state=42)
    arf.fit(X_train, y_train)


    y_pred = arf.predict(X_test)



    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-Squared Score:", r2)
    accuracy = arf.score(X, y)
    print('Model accuracy (R^2):', accuracy)

    claims

    print("_______________________________________________________________________")












def generate_feedback_ratings(claims):
    df3 = claims.copy()

    feedback_dict = {
        'Product is great!': 5,
        'Not satisfied with the product ': 1,
        'Fast delivery! ': 5,
        'Slow delivery': 3,
        'Great customer service!': 5,
        'Poor customer service': 2,
        'Product arrived damaged': 1,
        'Product exceeded my expectations': 5,
        'Product did not meet my expectations': 2,
        'Would recommend to others': 5,
        'Would not recommend to others': 1,
        'Excellent': 5,
        'Great': 5,
        'Good': 3,
        'Fair': 4,
        'Poor': 2,
        'Very poor': 1,
        'Satisfactory': 4,
        'Unsatisfactory': 2,
        'Outstanding': 4,
        'Terrible': 1,
        'Amazing': 5,
        'Average': 3,
        'Not bad': 3,
        'Could be better': 4,
        'Needs improvement': 3,
        'Excellent service!': 5,
        'Fast and efficient process.': 5,
        'Friendly and helpful staff.': 5,
        'Great value for money.': 5,
        'Smooth and seamless experience.': 5,
        'Prompt response to my claim.': 4,
        'Very satisfied with the outcome.': 4,
        'The repair work was top-notch.': 5,
        'Impressed with the level of professionalism.': 5,
        'Overall, a great experience.': 5,
        'I appreciate the attention to detail.': 5,
        'The customer service was outstanding.': 5,
        'Effortless and hassle-free process.': 5,
        'The claim was processed quickly.': 5,
        'Highly recommended!': 5,
        'Very impressed with the service.': 5,
        'The staff went above and beyond.': 5,
        'Thoroughly impressed with the outcome.': 4,
        'The repair work was excellent.': 5,
        'Thank you for the great service!': 5,
        "I couldn't be happier with the result.": 2,
        'The customer service exceeded my expectations.': 5,
        'The whole process was seamless.': 4,
        'I felt valued as a customer.': 5,
        'The level of care and attention was impressive.': 5,
        'Thank you for making it so easy.': 5,
        'The claim process was straightforward.': 5,
        'Excellent communication throughout the process.': 5,
        'The outcome was exactly what I was hoping for.': 5,
        'Great service, great staff!': 5,
        'The staff were friendly and professional.': 5,
        'The repair work was done to a high standard.': 4,
        'I appreciate the transparency and honesty.': 5,
        'The customer service team were fantastic.': 5,
        'The process was efficient and effective.': 5,
        'I was kept informed every step of the way.': 5,
        'The outcome exceeded my expectations.': 5,
        'Excellent value for money.': 5,
        'The staff were knowledgeable and helpful.': 5,
        'I would definitely recommend this service.': 4,
        'The repair work was completed quickly and efficiently.': 5}
    


    import random

    # create empty lists to store the feedbacks and ratings
    feedbacks = []
    ratings = []

    # loop through each row in the dataframe and randomly select a feedback and rating
    for i in range(len(df3)):
        feedback, rating = random.choice(list(feedback_dict.items()))
        feedbacks.append(feedback)
        ratings.append(rating)

    # add the feedback and rating columns to the dataframe
    df3['feedback'] = feedbacks
    df3['rating'] = ratings



    df3.head(60)







def train_decision_tree_classifier(claims):
    print("_______________________________________________________________________")

    claims1 = claims.copy()




    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier



    labelencoder = LabelEncoder()
    claims1['Part_ID'] = labelencoder.fit_transform(claims1['Part_ID'])
    claims1['Dealer_ID'] = labelencoder.fit_transform(claims1['Dealer_ID'])
    claims1['Repair_or_Replace'] = labelencoder.fit_transform(claims1['Repair_or_Replace'])




    X = claims1[["Dealer_ID","Part_ID"]]
    y = claims1["Repair_or_Replace"] # target variable


    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train, y_train)


    print("Decision Tree accuracy:", dtc.score(X_test, y_test))


    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)


    # evaluate Random Forest classifier
    print("Random Forest accuracy:", rfc.score(X_test, y_test))




    prediction = rfc.predict([[5,902]])

    print("_____________________________________________________________________________")
    return rfc


