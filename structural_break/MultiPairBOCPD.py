import math
import numpy as np
import pandas as pd

class MultiPairBOCPD:
    def __init__(self, pairs, bocpd_factory, cp_threshold=0.5):
        """
        pairs: list of tuples, e.g. [('AAPL','MSFT'), ('XOM','CVX')]
        bocpd_factory: callable -> fresh BOCPD instance
        cp_threshold: probability threshold for declaring CP
        """
        self.pairs = pairs
        self.cp_threshold = cp_threshold

        self.models = {p: bocpd_factory() for p in pairs}
        self.cp_probs = {p: [] for p in pairs}
        self.cp_flags = {p: [] for p in pairs}
        self.last_cp = {p: 0 for p in pairs}

    def update(self, pair, x_t, t):
        model = self.models[pair]
        model.update(x_t)

        cp_prob = model.change_probs[-1]
        self.cp_probs[pair].append(cp_prob)

        cp_flag = cp_prob > self.cp_threshold
        self.cp_flags[pair].append(cp_flag)

        if cp_flag:
            self.last_cp[pair] = t

        return cp_prob
