import logging
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("yfinance").propagate = False

from itertools import combinations
import yfinance as yf
from statsmodels.tsa.stattools import coint
import numpy as np
import pandas as pd
import requests
from io import StringIO
import datetime as dt
from pathlib import Path


class SP500PairSpread:
    """
    Selects stock pairs using a transparent, pre-period-only procedure.
    """

    def __init__(
        self,
        selection_start,
        selection_end,
        min_avg_volume=1e6,
        method="distance",          # "distance" or "cointegration"
        pairs_per_sector=3
    ):
        """
        Parameters
        ----------
        # universe_df : pd.DataFrame
        #     Must contain ['Symbol', 'Sector']
        selection_start : str
        selection_end : str
        min_avg_volume : float
        method : str
            'distance' or 'cointegration'
        pairs_per_sector : int
        """

        # self.universe_df = universe_df # Keep commented out
        self.selection_start = selection_start
        self.selection_end = selection_end
        self.min_avg_volume = min_avg_volume
        self.method = method
        self.pairs_per_sector = pairs_per_sector
        self.csv_path = "sp500_constituents.csv"
        self.csv_spread_cache_file = "sp500_spread_cache.csv"
        self.wiki_sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        self.selected_pairs = None
        self._get_sp500_symbols() # Initialize universe_df here

    # -----------------------------------------------------
    # Data utilities
    # -----------------------------------------------------

    def _get_sp500_symbols(self):

        _csv_path = Path(self.csv_path)

        if _csv_path.exists():
            print(f"Loading cached volume from {_csv_path}")
            # self.universe_df = pd.read_csv(_csv_path, index_col=0, parse_dates=True)
            self.universe_df = pd.read_csv(_csv_path)
        else:
            print("Fetching SP500 pairs from Wikipedia...")
            # url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }

            response = requests.get(self.wiki_sp500_url, headers=headers)
            response.raise_for_status()
            sp500 = pd.read_html(StringIO(response.text))[0]

            sp500 = sp500[['Symbol', 'GICS Sector']]
            sp500.columns = ['Symbol', 'Sector']
            sp500.to_csv("sp500_constituents.csv", index=False)
            self.universe_df = sp500

    def get_symbols(self):
        return self.universe_df['Symbol'].tolist()

    def _download_prices(self, symbols):
        data = yf.download(
            symbols,
            start=self.selection_start,
            end=self.selection_end,
            progress=False,
            auto_adjust=False,
            multi_level_index=False
        )['Adj Close']
        # Force DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()

        return data.dropna(axis=1)

    def _is_liquid(self, symbol):
        data = yf.download(
            symbol,
            start=self.selection_start,
            end=self.selection_end,
            progress=False,
            auto_adjust=False,
            multi_level_index=False
        )

        # No data at all
        if data is None or data.empty:
            return False

        # Volume column missing
        if 'Volume' not in data.columns:
            return False

        vol = data['Volume']

        # Force Series
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]

        avg_vol = float(vol.mean())

        return avg_vol >= float(self.min_avg_volume)

    def _has_full_price_history(self, symbol):
        data = yf.download(
            symbol,
            start=self.selection_start,
            end=self.selection_end,
            progress=False,
            auto_adjust=False,
            multi_level_index=False
        )

        return data is not None and not data.empty

    def _normalize_symbol(self, symbol):
        return symbol.replace('.', '-')

    # -----------------------------------------------------
    # Similarity metrics
    # -----------------------------------------------------

    @staticmethod
    def _distance_metric(p1, p2):
        p1n = p1 / p1.iloc[0]
        p2n = p2 / p2.iloc[0]
        return np.linalg.norm(p1n - p2n)

    @staticmethod
    def _cointegration_metric(p1, p2):
        score, _, _ = coint(p1, p2)
        return -score  # higher = stronger cointegration

    # -----------------------------------------------------
    # Core logic
    # -----------------------------------------------------

    def _filter_liquid_stocks(self):
        sector_map = {}

        for sector in self.universe_df['Sector'].unique():
            symbols = self.universe_df.loc[
                self.universe_df['Sector'] == sector, 'Symbol'
            ].tolist()

            liquid = []

            for s in symbols:
                try:
                    s = self._normalize_symbol(s)
                    if self._has_full_price_history(s) and self._is_liquid(s):
                        liquid.append(s)
                except Exception:
                    continue

            if len(liquid) >= 2:
                sector_map[sector] = liquid

        return sector_map

    def _select_pairs(self):
        """
        Run the full selection pipeline and freeze the pairs.
        """

        sector_stocks = self._filter_liquid_stocks()
        results = []

        for sector, stocks in sector_stocks.items():
            prices = self._download_prices(stocks)

            if prices is None or prices.empty or prices.shape[1] < 2:
                continue

            for s1, s2 in combinations(prices.columns, 2):
                p1, p2 = prices[s1], prices[s2]

                try:
                    if self.method == "distance":
                        score = self._distance_metric(p1, p2)
                    elif self.method == "cointegration":
                        score = self._cointegration_metric(p1, p2)
                    else:
                        raise ValueError("Unknown method")

                    results.append({
                        "pair": (s1, s2),
                        "sector": sector,
                        "score": score
                    })

                except Exception:
                    continue

        df = pd.DataFrame(results)

        # Ranking
        ascending = True if self.method == "distance" else False
        df = df.sort_values("score", ascending=ascending)

        # Sector-balanced selection
        selected = []
        for sector in df['sector'].unique():
            sector_df = df[df['sector'] == sector]
            selected.extend(
                sector_df.head(self.pairs_per_sector)['pair'].tolist()
            )

        self.selected_pairs = selected
        return selected

    # def get_selected_pairs(self):
    #     if self.selected_pairs is None:
    #         raise RuntimeError("Run select_pairs() first.")
    #     return self.selected_pairs

    # Calculate spread with hedge ratio of means
    def distance_spread(self):
        _csv_spread_path = Path(self.csv_spread_cache_file)

        if _csv_spread_path.exists():
            print(f"Loading cached volume from {_csv_spread_path}")
            # self._select_pairs()
            df_spread = pd.read_csv(_csv_spread_path, index_col=0, parse_dates=True)
        else:
            print("Calculating spread...")
            self._select_pairs()
            if self.selection_end is None:
                self.selection_end = dt.date.today()
            if self.selection_start is None:
                self.selection_start = end_date - dt.timedelta(days=365*2)  # ~2 years
            df = pd.DataFrame()

            # flatten unique tickers from tuple list
            tickers = sorted(set(t for pair in self.selected_pairs for t in pair))

            data = yf.download(tickers, start=self.selection_start, end=self.selection_end, auto_adjust=False, multi_level_index=False)['Adj Close']

            for first, second in self.selected_pairs:
                # skip pairs with missing data
                if first not in data.columns or second not in data.columns:
                    continue

                pair_data = data[[first, second]].dropna()
                if pair_data.empty:
                    continue

                hedge_ratio = pair_data[first].mean() / pair_data[second].mean()
                spread = pair_data[first] - hedge_ratio * pair_data[second]

                df[f'{first}-{second}'] = spread
                df_spread = df.dropna(axis=1)

            df_spread.to_csv(self.csv_spread_cache_file)

        return df_spread
