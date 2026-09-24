from flask import Flask, render_template, request
import pandas as pd
import joblib


app = Flask(__name__)


# Load trained model
model = joblib.load(
    "model/phone_price_prediction_model.pkl"
)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        brand = request.form["brand"]

        ram = float(
            request.form["ram"]
        )

        storage = float(
            request.form["storage"]
        )

        camera = float(
            request.form["camera"]
        )

        battery = float(
            request.form["battery"]
        )

        screen = float(
            request.form["screen"]
        )


        input_data = pd.DataFrame([
            {
                "Brand": brand,
                "RAM_GB": ram,
                "Storage_GB": storage,
                "Camera_MP": camera,
                "Battery_mAh": battery,
                "Screen_Size": screen
            }
        ])


        prediction = model.predict(
            input_data
        )[0]

        prediction = round(
            prediction,
            2
        )


    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)