#!/usr/bin/env python3

from flask import Flask, request, render_template
import joblib, json
import sklearn

# print(joblib.__version__)
# print(sklearn.__version__)
# exit()

app = Flask(__name__)

model = joblib.load("churn_prediction.pkl")
transformer = joblib.load("transformer.pkl")

# curl --location 'http://127.0.0.1:5000/churn_predictions' \
# --header 'Content-Type: application/json' \
# --data '{
#     "data": {
#         "credit_score": 678,
#         "geography": "France",
#         "gender": "Female",
#         "age": 45,
#         "balance": "545253",
#         "number_of_products": 2,
#         "is_active_member": 0
#     }
# }
# '
@app.route("/churn_predictions", methods=["GET", "POST"])
def create_churn_prediction():
    request_hash = json.loads(request.data.decode())
    data_point = request_hash['data'].values()
    cust = transformer.transform([list(data_point)])
    out = model.predict(cust)
    response_hash = {}
    response_hash['data'] = { "prediction": str(out[0]) }
    return json.dumps(response_hash)

# @app.route("/", methods=["GET"])
# def get_home():
#   return render_template("index.html")

# @app.route("/", methods=["POST"])
# def post_home():
#   request_hash = dict(request.form)
#   cust = transformer.transform([list(request_hash.values())])
#   out = model.predict(cust)
#   response_hash = {}
#   response_hash['prediction'] = str(out[0])
#   return json.dumps(response_hash)

if __name__ == "__main__":
    app.run(debug=True)
