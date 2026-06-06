from flask import Flask, render_template, request, jsonify
from helpers import *
from yfinance import download
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd

app = Flask(__name__)

def load_data():
    df_qqq = download("QQQ", period="max", interval="1d")
    df_vix = download("^VIX", period="max", interval="1d")
    df_qqq.reset_index(inplace=True)
    df_vix.reset_index(inplace=True)
    if isinstance(df_qqq.columns, pd.MultiIndex):
        df_qqq.columns = df_qqq.columns.droplevel(1)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.droplevel(1)
    df_qqq = df_qqq[["Date", "Close", "Volume"]]
    vix_df = df_vix[["Date", "Close"]].rename(columns={"Close": "VIX"})
    df = df_qqq.merge(vix_df, on="Date", how="left")
    df["VIX"] = df["VIX"].ffill()
    return df

DF_CACHE = load_data()

def refresh_data():
    global DF_CACHE
    DF_CACHE = load_data()
    print("Data refreshed")

scheduler = BackgroundScheduler()
scheduler.add_job(refresh_data, 'cron', hour='9,17') 
scheduler.start()


@app.route('/data', methods=['POST'])
def data_route():
    df = DF_CACHE[['Date', 'Close']].copy()
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return jsonify(df.to_dict(orient='records'))

@app.route('/predict', methods=['POST'])
def predict_route():
    look_back = 60
    X_mag, X_dir, dates = preprocess_data(DF_CACHE, look_back)
    predictions = predict(X_mag, X_dir)

    response = {
        "dates": [d.strftime('%Y-%m-%d') for d in dates.tolist()],  
        "predictions": {
            "move_3d":  predictions["move_3d"].tolist(),
            "move_5d":  predictions["move_5d"].tolist(),
            "move_10d": predictions["move_10d"].tolist(),
            "dir_3d":  [1 if p > 0.5 else -1 for p in predictions["prob_3d"]],
            "dir_5d":  [1 if p > 0.5 else -1 for p in predictions["prob_5d"]],
            "dir_10d": [1 if p > 0.5 else -1 for p in predictions["prob_10d"]],
        }
    }
    return jsonify(response)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():

    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)