# automation_script

import pandas as pd
from data_processing import preprocess_data, encode_labels
from modeling import usecase_1, predict_fail_parts_by_location, predict_fail_parts_by_mileage, predict_fail_parts_by_age, generate_feedback_ratings, train_decision_tree_classifier

def main():
    #read the data
    data1 = pd.read_excel('D:/DS_Project/Dataset.xlsx', sheet_name=['Dealers', 'Customers', 'Claims', 'Parts', 'Transactions', 'Vendors'])

    #ddata preprocessing
    dealers, customers, claims, parts, transactions, vendors = preprocess_data(data1)

    
    dealers = encode_labels(dealers)
    customers = encode_labels(customers)
    claims = encode_labels(claims)
    parts = encode_labels(parts)
    transactions = encode_labels(transactions)
    vendors = encode_labels(vendors)

    # usecase 1 analysis
    usecase_1(claims)

    print("__________________________________________________________")

    predict_fail_parts_by_location(claims, dealers)

    print("__________________________________________________________")

    predict_fail_parts_by_mileage(claims)

    print("__________________________________________________________")

    predict_fail_parts_by_age(claims, parts)

    print("__________________________________________________________")

    generate_feedback_ratings(claims)






    rfc_model = train_decision_tree_classifier(claims)

    prediction = rfc_model.predict([[5, 902]])
    print("RFC Prediction:", prediction)

if __name__ == "__main__":
    main()
