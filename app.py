import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import GradientBoostingRegressor
from flask import Flask, request, jsonify

# Logging & Monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data Processing Module
class DataProcessor:
    def __init__(self, sales_data, inventory_data, external_factors):
        self.sales_data = sales_data
        self.inventory_data = inventory_data
        self.external_factors = external_factors
    
    def prepare_data(self):
        # Merge all data sources
        data = pd.merge(self.sales_data, self.inventory_data, on='product_id')
        data = pd.merge(data, self.external_factors, on='date', how='left')
        data.fillna(0, inplace=True)
        return data

# AI Models (Demand Forecasting with Advanced ML)
class DemandForecaster:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

# Multi-Agent System
class RetailAgent:
    def __init__(self, name, inventory):
        self.name = name
        self.inventory = inventory
    
    def check_stock(self, product_id):
        stock = self.inventory.get(product_id, 0)
        logging.info(f"Checking stock for {product_id}: {stock}")
        return stock
    
    def order_stock(self, product_id, quantity):
        self.inventory[product_id] = self.inventory.get(product_id, 0) + quantity
        logging.info(f"Ordered {quantity} units of {product_id}")
        return f"Ordered {quantity} units of {product_id}"

# API Layer
app = Flask(__name__)
retail_agent = RetailAgent("Store_1", {"P1": 20, "P2": 50})

data_processor = DataProcessor(
    sales_data=pd.DataFrame({'product_id': ['P1', 'P2'], 'sales': [100, 150], 'date': ['2024-03-01', '2024-03-01']}),
    inventory_data=pd.DataFrame({'product_id': ['P1', 'P2'], 'stock': [20, 50]}),
    external_factors=pd.DataFrame({'date': ['2024-03-01'], 'holiday': [1], 'weather': [0.8]})
)

demand_forecaster = DemandForecaster()
data = data_processor.prepare_data()
X_train = data[['stock', 'holiday', 'weather']]
y_train = data['sales']
demand_forecaster.train(X_train, y_train)

@app.route('/check_stock', methods=['GET'])
def check_stock():
    product_id = request.args.get('product_id')
    stock = retail_agent.check_stock(product_id)
    return jsonify({"product_id": product_id, "stock": stock})

@app.route('/order_stock', methods=['POST'])
def order_stock():
    data = request.get_json()
    if 'product_id' not in data or 'quantity' not in data:
        return jsonify({"error": "Missing product_id or quantity"}), 400

    response = retail_agent.order_stock(data['product_id'], data['quantity'])
    return jsonify({"message": response})


@app.route('/predict_demand', methods=['GET'])
def predict_demand():
    product_id = request.args.get('product_id')
    input_data = np.array([[retail_agent.check_stock(product_id), 1, 0.8]])  # Example external factors
    prediction = demand_forecaster.predict(input_data)[0]
    logging.info(f"Predicted demand for {product_id}: {prediction}")
    return jsonify({"product_id": product_id, "predicted_sales": prediction})

if __name__ == '__main__':
    logging.info("Starting Retail Inventory AI System...")
    app.run(debug=True)
