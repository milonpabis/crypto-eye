from models_container.EstimatorsBTC import EstimatorsBTC
import datetime as dt

if __name__ == "__main__":
    print("--------- Loading and preprocessing data ---------")
    start = dt.datetime.now()
    btc = EstimatorsBTC()
    #btc.update_performance(days_back=250)
    btc.calculate_performance("RandomForest")
    end = dt.datetime.now()
    print(f"--------- Time taken: {end-start} ---------")


# TODO:
# model performance for different time periods
# function calcultaing model performance
