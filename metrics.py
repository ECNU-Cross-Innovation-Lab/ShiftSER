class IEMOCAP_Meter:
    """Computes and stores the current best value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.WA = 0.
        self.UA = 0.
        self.pred_WA = []
        self.label_WA = []
        self.pred_UA = []
        self.label_UA = []

    def update(self, WA, UA, pred, label):
        if WA > self.WA:
            self.WA = WA
            self.pred_WA = pred
            self.label_WA = label
        if UA > self.UA:
            self.UA = UA
            self.pred_UA = pred
            self.label_UA = label

