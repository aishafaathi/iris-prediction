from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    s1 = float(request.form['sepal_length'])
    s2 = float(request.form['sepal_width'])
    p1 = float(request.form['petal_length'])
    p2 = float(request.form['petal_width'])

    features = [[s1, s2, p1, p2]]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_sc .aled)[0]
    flower = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    result = flower[prediction]
    print("****Prediction is", prediction)

    return render_template('index.html', prediction=result) 
if __name__ == '__main__':
    app.run(debug=True)
