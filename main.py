from models_container.EstimatorsBTC import EstimatorsBTC
import datetime as dt
import matplotlib.pyplot as plt
from app import app

DEBUG = False
APP_TEST = True
est = ["RandomForest", "AdaBoost", "GradientBoost"]


if __name__ == "__main__":

    if APP_TEST:
        app.run(debug=True)



    else:

        if not DEBUG:
            print("--------- Loading and preprocessing data ---------")
            start = dt.datetime.now()
            btc = EstimatorsBTC()
            btc.update_predictions(days_back=250)
            btc.update_performance("RandomForest")
            btc.update_performance("AdaBoost")
            btc.update_performance("GradientBoost")

            print(btc.get_prediction_today())
            print(btc.modelDB.get_model_performance("RandomForest").tail())
            end = dt.datetime.now()
            print(f"--------- Time taken: {end-start} ---------")

        else:
            import yfinance as yf

            btc = yf.Ticker("BTC-USD")
            btc = btc.history(start=None, end=None, period="max")
            print(btc.tail())



# TODO:
# model performance for different time periods
# function calcultaing model performance DONE
# if there is already a predicted value in db - return instead of predicting again DONE