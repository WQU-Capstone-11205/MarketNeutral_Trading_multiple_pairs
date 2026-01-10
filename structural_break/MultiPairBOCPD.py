import math
import numpy as np
import pandas as pd

class MultiPairBOCPD:
    def __init__(self, pairs, hazard, distribution_cls):
        """
        pairs: list of pair names, e.g. [("AAPL","MSFT"), ("XOM","CVX")]
        hazard: hazard function
        distribution_cls: callable returning a fresh likelihood model
        """
        self.models = {}
        self.cp_probs = {}
        self.cp_flags = {}

        for pair in pairs:
            self.models[pair] = BOCPD(
                hazard=hazard,
                distribution=distribution_cls()
            )
            self.cp_probs[pair] = []
            self.cp_flags[pair] = []

    def update(self, pair, x_t):
        """
        Update BOCPD for a single pair
        """
        model = self.models[pair]
        model.update(x_t)

        # Probability of change point at time t
        cp_prob = model.change_probs[-1]
        self.cp_probs[pair].append(cp_prob)

        # Binary flag (optional)
        self.cp_flags[pair].append(cp_prob > 0.5)

        return cp_prob
