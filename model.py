import pandas as pd
import joblib

vectorizer = joblib.load('dumped/vectorizer')
model = joblib.load('dumped/model')

def get_prediction(text):
    my_tf = vectorizer.transform(pd.DataFrame([text], columns=['text'])['text'])

    return model.predict_proba(my_tf)[0][1]
