from models_container.EstimatorsBTC import EstimatorsBTC
import datetime as dt

DEBUG = False
est = ["RandomForest", "AdaBoost", "GradientBoost"]


if __name__ == "__main__":

    if not DEBUG:
        print("--------- Loading and preprocessing data ---------")
        start = dt.datetime.now()
        btc = EstimatorsBTC()
        #btc.update_predictions(days_back=250)
        for e in est:
            btc.calculate_performance(e)
        #btc.calculate_performance("RandomForest")
        #btc.calculate_performance("AdaBoost")
        #btc.calculate_performance("GradientBoost")
        end = dt.datetime.now()
        print(f"--------- Time taken: {end-start} ---------")

    else:
        import yfinance as yf

        btc = yf.Ticker("BTC-USD")
        btc = btc.history(start=None, end=None, period="max")
        print(btc.tail())



# TODO:
# model performance for different time periods
# function calcultaing model performance
# if there is already a predicted value in db - return instead of predicting again