import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('my_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index_1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index_1.html', prediction_text='Wine Quality should be  {}'.format(output))

if __name__ == "__main__":
    app.run(debug=False,use_reloader=False)