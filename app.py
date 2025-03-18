from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form.get(feature)) for feature in ['brand', 'year', 'horsepower', 'mileage']]
    prediction = model.predict([features])[0]
    return render_template('index.html', prediction_text=f'Predicted Car Price: ₹{prediction:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
