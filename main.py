from models_container.EstimatorsBTC import EstimatorsBTC
import datetime as dt

if __name__ == "__main__":
    print("--------- Loading and preprocessing data ---------")
    start = dt.datetime.now()
    btc = EstimatorsBTC()
    btc.update_performance(days_back=200)
    end = dt.datetime.now()
    print(f"--------- Time taken: {end-start} ---------")


# TODO:
# database with evaluation for past 150 days + method to update missing days
# method to track the performance of the model