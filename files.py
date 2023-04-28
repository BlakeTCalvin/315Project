import pandas as pd

# load datasets
aisles = pd.read_csv('Files/aisles.csv')
departments = pd.read_csv('Files/departments.csv')
order_products_prior = pd.read_csv('Files/order_products__prior.csv')
order_products_train = pd.read_csv('Files/order_products__train.csv')
orders = pd.read_csv('Files/orders.csv')
products = pd.read_csv('Files/products.csv')