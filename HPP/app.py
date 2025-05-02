from flask import Flask, render_template, request
import pickle
import numpy as np
import json

app = Flask(__name__)

# Load model and columns
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    __data_columns = json.load(f)['data_columns']
    __locations = __data_columns[3:]

# Prediction function
def get_estimated_price(input_json):
    try:
        location = input_json['location'].strip().lower()
        loc_index = __data_columns.index(location)
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = float(input_json['sqft'])
    x[1] = float(input_json['bath'])
    x[2] = float(input_json['bhk'])

    if loc_index >= 0:
        x[loc_index] = 1

    try:
        result = round(model.predict([x])[0], 2)
        return result
    except:
        return None

# Routes
@app.route('/')
def index():
    return render_template('index.html', locations=__locations)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_json = {
            "location": request.form['sLocation'],
            "sqft": request.form['Squareft'],
            "bhk": request.form['uiBHK'],
            "bath": request.form['uiBathrooms']
        }

        try:
            result = get_estimated_price(input_json)
            if result is None:
                result = "Sorry, we could not estimate the price for the given inputs."
            else:
                result = f"{round(result / 100, 2)} Crore" if result > 100 else f"{result} Lakhs"
        except:
            result = "Sorry, an error occurred during prediction."

        return render_template('predict.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
