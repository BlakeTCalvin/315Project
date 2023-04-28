from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
import files

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

