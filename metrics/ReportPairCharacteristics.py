import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


class ReportPairCharacteristics:
    """
    Reports cross-sectional characteristics of selected pairs to demonstrate
    heterogeneity (cointegration strength, volatility, stability, distance-based similarity).
    """

    def __init__(self, spread_df, price_df):
        """
        Parameters
        ----------
        spread_df : pd.DataFrame
            time × pairs (e.g. AAPL-MSFT)
        price_df : pd.DataFrame
            time × tickers (Adj Close)
        """
        self.spread_df = spread_df
        self.price_df = price_df
        self.characteristics_df = None

    # ---------------------------------------------------
    # Pair-level diagnostics
    # ---------------------------------------------------

    @staticmethod
    def cointegration_pvalue(p1, p2):
        _, pval, _ = coint(p1, p2)
        return pval

    @staticmethod
    def spread_volatility(spread):
        return spread.std()

    @staticmethod
    def mean_reversion_half_life(spread):
        """
        Estimates half-life via AR(1) approximation.
        """
        delta = spread.diff().dropna()
        lagged = spread.shift(1).dropna()

        if len(delta) < 10:
            return np.nan

        beta = np.polyfit(lagged, delta, 1)[0]
        return -np.log(2) / beta if beta < 0 else np.inf

    @staticmethod
    def normalized_price_distance(p1, p2):
        """
        Distance metric used in pairs selection literature.
        """
        p1n = p1 / p1.iloc[0]
        p2n = p2 / p2.iloc[0]
        return np.linalg.norm(p1n - p2n)

    # ---------------------------------------------------
    # Core report
    # ---------------------------------------------------

    def compute_characteristics(self):
        rows = []

        for pair in self.spread_df.columns:
            s1, s2 = pair.split("-")

            spread = self.spread_df[pair].dropna()

            if s1 not in self.price_df.columns or s2 not in self.price_df.columns:
                continue

            p1 = self.price_df[s1].loc[spread.index]
            p2 = self.price_df[s2].loc[spread.index]

            half_life = self.mean_reversion_half_life(spread)

            if np.isinf(half_life):
                self.spread_df.drop(columns=[pair], inplace=True)
                continue
            
            rows.append({
                "pair": pair,
                "coint_pvalue": self.cointegration_pvalue(p1, p2),
                "norm_price_dist": self.normalized_price_distance(p1, p2),
                "spread_vol": self.spread_volatility(spread),
                "half_life": half_life
            })

        self.characteristics_df = (
            pd.DataFrame(rows)
            .set_index("pair")
            #.sort_values("cointegration_pvalue")
        )

        return self.characteristics_df

    # ---------------------------------------------------
    # Dispersion summary
    # ---------------------------------------------------

    def dispersion_summary(self):
        if self.characteristics_df is None:
            self.compute_characteristics()

        return self.characteristics_df.describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        )
