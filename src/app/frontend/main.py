import os

from flask import Flask, render_template

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/")
app = Flask(__name__)

@app.route("/")
def get_home():
    predict_endpoint = API_URL + "predict"
    return render_template("index.html", predict_endpoint=predict_endpoint)
