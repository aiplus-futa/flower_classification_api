# Importing Necessary Libraries
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Creating the flask app object
app = Flask(__name__)

# Load the serialized model
model = pickle.load(open("model_pipeline.pkl", "rb"))

# Create a route that goes to /, /home and /index
@app.route("/")
@app.route("/home")
@app.route("/index")
def index():
    return jsonify(message="Hello World")

@app.route("/predict/<X>")
def predict(X):
    X = np.fromstring(X, sep=",")
    X = X.reshape(1, X.shape[0])
    y_predicted = model.predict(X)
    print(y_predicted)
    labels = {0 : 'iris-setosa',
              1 : 'iris-versicolor',
              2 : 'iris-virginica'}

    y_predicted = np.vectorize(labels.__getitem__)(y_predicted)
    return jsonify(answer=str(y_predicted[0]))

if __name__ == "__main__":
    app.run(debug=True)