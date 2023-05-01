from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
import pandas as pd
import numpy as np
import gc
import files

def main():
    # merging datasets into one
    df = files.order_products_train.merge(files.products[['product_id', 'department_id']], on = 'product_id')
    df = df.merge(files.departments[['department_id', 'department']], on = 'department_id')
    df = df.merge(files.orders[['order_id', 'user_id']], on = 'order_id')

    # only the important features and reordered as the label
    features = df[['user_id', 'order_id', 'product_id']]
    label = df['reordered']

    # splitting the features and label 
    # (not sure if this is correct since all of order_products_train should probably be trained and not test)
    xTrain, xTest, yTrain, yTest = train_test_split(features, label, test_size = 1, random_state = 42)

    # creating classifier and fitting it
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(xTrain, yTrain)

    # check performance on test data
    # test = df[['user_id', 'eval_set' == 'test']]

    # make predictions on users
    for i in range(100):
        tempUserID = i
        userData = df[df['user_id'] == tempUserID][['user_id', 'product_id']]
        prediction = classifier.predict(userData)
    
def features():
    # read in orders and prior order products csv's
    orders = pd.read_csv('Files/orders.csv')
    prior_orders = pd.read_csv('Files/order_products__prior.csv')

    # merge the two dataframes. Merging directly to orders gets rid of product_id
    extra = orders.merge(prior_orders, on='order_id', how='left')

    # create new dataframe that will be used to show how often a user purhcases a product
    user_product_purchases = extra.groupby(['user_id', 'product_id'])[['order_id']].count()
    user_product_purchases.columns = ['total_purchased']
    user_product_purchases = user_product_purchases.reset_index()

    items = user_product_purchases[user_product_purchases.total_purchased == 1].groupby('product_id')[['total_purchased']].count()
    items.columns = ['']
    
    # garbage collect
    gc.collect()

if __name__ == '__main__':
    features()

# 1.) dataframe for user and each item and total orders, reordered or not
# 2.) dataframe of user and order id
# 3.) dataframe of each order and the items in the order
# 4.) find the number of times user ordered each item. Calculate based on total orders with if it was reasonably reordered.
# 5.) make predictions
