class EvalMetrics:
    def __init__(self, tp, fp, tn, fn):
        ''' Initialize Confusion Matrix Values'''
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        
    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
    
    def get_precision(self):
        return self.tp / (self.tp + self.fp)

    def get_recall(self):
        return self.tp / (self.tp + self.fn)

    def get_f1_score(self):
        ''' Weighted average of Precision and Recall'''
        
        precision = self.get_precision()
        recall = self.get_recall()
        
        return (2*precision*recall) / (precision+recall)