import numpy as np
import math

# -------------------------
# Utility: running stats
# -------------------------
class RunningMeanStd:
    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x):
        x = np.asarray(x)
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.size
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = max(new_var, 1e-6)
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (math.sqrt(self.var) + 1e-8)
