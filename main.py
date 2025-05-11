from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

class TextData(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: TextData):
    vec = vectorizer.transform([data.text])
    pred = model.predict(vec)
    return {"sentiment": "positive" if pred[0] == 1 else "negative"}
