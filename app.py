from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # required for server
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os

app = Flask(__name__)

# Load model once (important for performance)
model = load_model("Latest_stock_price_model.keras")

@app.route("/", methods=["GET", "POST"])
def index():
    stock = "GOOG"
    plot_path = None

    if request.method == "POST":
        stock = request.form.get("stock")

        end = datetime.now()
        start = datetime(end.year - 20, end.month, end.day)
        google_data = yf.download(stock, start, end)

        splitting_len = int(len(google_data) * 0.7)
        x_test = pd.DataFrame(google_data['Close'][splitting_len:])
        x_test.columns = ['Close']

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test[['Close']])

        x_data, y_data = [], []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
            y_data.append(scaled_data[i])

        x_data, y_data = np.array(x_data), np.array(y_data)

        predictions = model.predict(x_data)
        inv_pre = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)

        ploting_data = pd.DataFrame({
            'original': inv_y_test.reshape(-1),
            'predicted': inv_pre.reshape(-1)
        }, index=google_data.index[splitting_len + 100:])

        # Create plot
        if not os.path.exists("static"):
            os.makedirs("static")

        plot_path = "static/plot.png"
        plt.figure(figsize=(15, 6))
        plt.plot(pd.concat([google_data.Close[:splitting_len + 100], ploting_data], axis=0))
        plt.legend(["Data-Not Used", "Original Test Data", "Predicted Data"])
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return render_template("index.html", stock=stock, plot_path=plot_path)


if __name__ == "__main__":
    app.run()
