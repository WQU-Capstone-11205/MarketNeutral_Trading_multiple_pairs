import numpy as np
import random
from typing import Dict, List, Any
from itertools import product

class hyperparameter_tuner:
    def __init__(self):
        pass

    def gridsearch_from_space(self, space):
        keys = list(space.keys())
        values = list(space.values())
        for combo in product(*values):
            yield dict(zip(keys, combo))

    def sample_from_space(self, space, n=1):
        keys = list(space.keys())
        for _ in range(n):
            yield {k: random.choice(space[k]) for k in keys}

    def split_tvt(self, spreads, train_frac=0.6, val_frac=0.2):
        T = len(spreads) # Corrected: spreads is a DataFrame, len(DataFrame) gives number of rows
        t1 = int(T * train_frac)
        t2 = int(T * (train_frac + val_frac))

        train = {p: s.iloc[:t1] for p, s in spreads.items()}
        val   = {p: s.iloc[t1:t2] for p, s in spreads.items()}
        test  = {p: s.iloc[t2:T] for p, s in spreads.items()}

        return train, val, test

    def walk_forward_splits(self, spreads, train_len, val_len, step):
        T = len(spreads)
        t = 0

        while t + train_len + val_len < T:
            train = {p: s.iloc[t:t+train_len] for p, s in spreads.items()}
            val   = {p: s.iloc[t+train_len:t+train_len+val_len] for p, s in spreads.items()}
            yield train, val
            t += step
