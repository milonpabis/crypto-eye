


class PerformanceBatch:

    def __init__(self, recall: float, precision: float, accuracy: float, specificity: float, neg_pred_value: float):
        self.recall = recall
        self.precision = precision
        self.accuracy = accuracy
        self.specificity = specificity
        self.neg_pred_value = neg_pred_value

    
    def get_data(self):
        return self.recall, self.precision, self.accuracy, self.specificity, self.neg_pred_value



class PerformanceWindows:


    def __init__(self, estimator: str,  date: str, batch_total: PerformanceBatch, batch_7: PerformanceBatch, batch_14: PerformanceBatch, batch_30: PerformanceBatch):
        self.estimator = estimator
        self.date = date
        self.batch_total = batch_total
        self.batch_7 = batch_7
        self.batch_14 = batch_14
        self.batch_30 = batch_30


    def get_data(self):
        return *self.batch_total.get_data(), *self.batch_7.get_data(), *self.batch_14.get_data(), *self.batch_30.get_data()
    
    def get_estimator(self):
        return self.estimator
    
    def get_date(self):
        return self.date
        





    