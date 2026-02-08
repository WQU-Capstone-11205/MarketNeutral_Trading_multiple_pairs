import numpy as np
import pandas as pd

class ReportDistributionalResults:
    """
    Computes distributional performance statistics across pairs:
    - median performance
    - interquartile range (IQR)
    - worst-decile drawdowns
    """

    def __init__(self, spread_df, _mean_train, _std_train):
        """
        Parameters
        ----------
        spread_df   : pd.DataFrame
            time × pairs spread series
        _mean_train : mean of training data
        _std_train  : Standard deviation of 
            training data
        """
        self.spread_df = spread_df
        self.mean_train = _mean_train
        self.std_train = _std_train
        self.pnl_df = None

    # -----------------------------
    # Core building blocks
    # -----------------------------

    @staticmethod
    def compute_pair_pnl(spread_test, mean_train, std_train):
        """
        Mean-reversion PnL with unit exposure.
        """
        z = (spread_test - mean_train) / std_train
        position = -np.sign(z)
        pnl = position.shift(1) * spread_test.diff()
        return pnl.dropna()

    @staticmethod
    def max_drawdown(cum_pnl):
        running_max = cum_pnl.cummax()
        drawdown = running_max - cum_pnl
        return drawdown.max()

    # -----------------------------
    # Pipeline steps
    # -----------------------------

    def build_pnl_matrix(self):
        pnl_df = pd.DataFrame(index=self.spread_df.index)

        for pair in self.spread_df.columns:
            pnl_df[pair] = self.compute_pair_pnl(self.spread_df[pair], self.mean_train[pair], self.std_train[pair])

        self.pnl_df = pnl_df.dropna(how="all")
        return self.pnl_df

    def distributional_stats(self):
        if self.pnl_df is None:
            self.build_pnl_matrix()

        # Terminal PnL per pair
        terminal_pnl = self.pnl_df.cumsum().iloc[-1]

        # Max drawdown per pair
        pair_drawdowns = self.pnl_df.cumsum().apply(self.max_drawdown)

        stats = {
            "median_performance": terminal_pnl.median(),
            "p25_performance": terminal_pnl.quantile(0.25),
            "p75_performance": terminal_pnl.quantile(0.75),
            "interquartile_range": (
                terminal_pnl.quantile(0.75) -
                terminal_pnl.quantile(0.25)
            ),
            "worst_decile_drawdown": pair_drawdowns.quantile(0.90)
        }

        return stats
