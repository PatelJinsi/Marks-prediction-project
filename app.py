from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", 'rb'))

@app.route('/')
def home():
    return "API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    study_hours = data["study_hours"]
    attendance = data["attendance"]

    # Pass input as DataFrame for correct feature names
    input_data = pd.DataFrame([[study_hours, attendance]], columns=["study_hours", "attendance"])
    result = model.predict(input_data)

    # Access first element
    return jsonify({
        "marks": float(result[0])
    })

if __name__ == '__main__':
    app.run(debug=True)