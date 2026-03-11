from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "credit_project"

scaler, pca, model = joblib.load("credit_model.pkl")


@app.route('/')
def home():

    category = session.pop("category", None)
    status = session.pop("status", None)
    limit = session.pop("limit", None)
    risk = session.pop("risk", None)

    return render_template(
        "index.html",
        category=category,
        status=status,
        limit=limit,
        risk=risk
    )


@app.route('/predict', methods=['POST'])
def predict():

    balance = float(request.form['balance'])
    purchases = float(request.form['purchases'])
    cash = float(request.form['cash'])
    credit = float(request.form['credit'])
    payments = float(request.form["payments"])

    data = np.array([[balance, purchases, cash, credit, payments]])

    scaled = scaler.transform(data)
    reduced = pca.transform(scaled)

    cluster = model.predict(reduced)[0]

    if cluster == 0:
        session["category"] = "Low Spending Customer"
        session["status"] = "Basic Credit Card Approved"
        session["limit"] = "₹50,000"
        session["risk"] = "Low"

    elif cluster == 1:
        session["category"] = "Premium Customer"
        session["status"] = "Platinum Card Approved"
        session["limit"] = "₹5,00,000"
        session["risk"] = "Very Low"

    elif cluster == 2:
        session["category"] = "Moderate Customer"
        session["status"] = "Standard Card Approved"
        session["limit"] = "₹1,50,000"
        session["risk"] = "Medium"

    else:
        session["category"] = "High Risk Customer"
        session["status"] = "Application Rejected"
        session["limit"] = "Not Eligible"
        session["risk"] = "High"

    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)