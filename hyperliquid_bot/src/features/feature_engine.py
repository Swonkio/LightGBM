"""
Real-time feature engineering for Hyperliquid perpetuals trading.

Computes 45+ features across multiple categories:
- Orderbook microstructure
- Volume and order flow
- Volatility (realized, Parkinson, Garman-Klass)
- Funding and open interest
- Technical indicators
- Temporal encoding
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Real-time feature computation engine optimized for speed."""

    def __init__(self, lookback_windows: List[int] = [5, 15, 30, 60, 240]):
        self.lookback_windows = lookback_windows
        self.feature_cache = {}

        # Rolling buffers for efficient computation
        self.price_buffer = deque(maxlen=max(lookback_windows) * 2)
        self.volume_buffer = deque(maxlen=max(lookback_windows) * 2)
        self.trade_buffer = deque(maxlen=1000)
        self.orderbook_buffer = deque(maxlen=100)

    def compute_all_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute all 45+ features from raw market data.

        Args:
            df: Polars DataFrame with columns: timestamp, open, high, low, close, volume

        Returns:
            DataFrame with all features appended
        """
        if df.is_empty():
            return df

        # Ensure datetime type
        if df["timestamp"].dtype != pl.Datetime:
            df = df.with_columns(
                pl.col("timestamp").cast(pl.Datetime)
            )

        # 1. Orderbook microstructure features
        df = self._add_orderbook_features(df)

        # 2. Volume and order flow features
        df = self._add_volume_features(df)

        # 3. Volatility features
        df = self._add_volatility_features(df)

        # 4. Technical indicators
        df = self._add_technical_indicators(df)

        # 5. Temporal features
        df = self._add_temporal_features(df)

        # 6. Lag features
        df = self._add_lag_features(df)

        # 7. Rolling statistics
        df = self._add_rolling_stats(df)

        return df

    def _add_orderbook_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute orderbook microstructure features."""
        features = df

        # Micro-price (weighted mid)
        if "best_bid" in df.columns and "best_ask" in df.columns:
            features = features.with_columns([
                (
                    (pl.col("best_bid") * pl.col("ask_size") +
                     pl.col("best_ask") * pl.col("bid_size")) /
                    (pl.col("bid_size") + pl.col("ask_size") + 1e-10)
                ).alias("micro_price")
            ])

            # Spread in basis points
            features = features.with_columns([
                (
                    (pl.col("best_ask") - pl.col("best_bid")) /
                    ((pl.col("best_ask") + pl.col("best_bid")) / 2) * 10000
                ).alias("spread_bps")
            ])

            # Order flow imbalance
            features = features.with_columns([
                (
                    (pl.col("bid_size") - pl.col("ask_size")) /
                    (pl.col("bid_size") + pl.col("ask_size") + 1e-10)
                ).alias("orderflow_imbalance")
            ])

            # Book depth ratio (top level)
            features = features.with_columns([
                (pl.col("bid_size") / (pl.col("ask_size") + 1e-10)).alias("book_depth_ratio")
            ])

        return features

    def _add_volume_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute volume and trade flow features."""
        features = df

        if "volume" not in df.columns:
            return features

        # Volume delta (assume buy/sell split based on close vs open)
        features = features.with_columns([
            pl.when(pl.col("close") > pl.col("open"))
            .then(pl.col("volume"))
            .otherwise(-pl.col("volume"))
            .alias("volume_delta")
        ])

        # Cumulative volume delta
        features = features.with_columns([
            pl.col("volume_delta").cum_sum().alias("cumulative_delta")
        ])

        # Volume moving average ratio
        for window in [15, 30, 60]:
            features = features.with_columns([
                (
                    pl.col("volume") /
                    (pl.col("volume").rolling_mean(window) + 1e-10)
                ).alias(f"volume_ma_ratio_{window}")
            ])

        # Buy/sell volume ratio (estimated)
        features = features.with_columns([
            pl.when(pl.col("close") > pl.col("open"))
            .then(1.0)
            .otherwise(0.0)
            .rolling_mean(15)
            .alias("buy_sell_ratio_15")
        ])

        return features

    def _add_volatility_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute various volatility estimators."""
        features = df

        # Realized volatility (close-to-close)
        for window in [15, 60]:
            returns = np.log(features["close"] / features["close"].shift(1))
            features = features.with_columns([
                (returns.rolling_std(window) * np.sqrt(window)).alias(f"realized_vol_{window}")
            ])

        # Parkinson volatility (high-low range estimator)
        hl_ratio = np.log(features["high"] / features["low"]) ** 2
        for window in [15, 60]:
            features = features.with_columns([
                (
                    (hl_ratio.rolling_mean(window) / (4 * np.log(2))) ** 0.5
                ).alias(f"parkinson_vol_{window}")
            ])

        # Garman-Klass volatility
        oc = np.log(features["close"] / features["open"]) ** 2
        gk_vol = 0.5 * hl_ratio - (2 * np.log(2) - 1) * oc

        for window in [15, 60]:
            features = features.with_columns([
                (gk_vol.rolling_mean(window) ** 0.5).alias(f"garman_klass_vol_{window}")
            ])

        return features

    def _add_technical_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute standard technical indicators."""
        features = df

        # RSI
        for window in [14, 28]:
            delta = features["close"].diff()
            gain = delta.clip(lower_bound=0)
            loss = -delta.clip(upper_bound=0)

            avg_gain = gain.rolling_mean(window)
            avg_loss = loss.rolling_mean(window)

            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            features = features.with_columns([rsi.alias(f"rsi_{window}")])

        # Bollinger Bands distance
        for window in [20, 40]:
            ma = features["close"].rolling_mean(window)
            std = features["close"].rolling_std(window)
            bb_upper = ma + 2 * std
            bb_lower = ma - 2 * std

            # Normalized distance from bands
            bb_dist = (features["close"] - ma) / (std + 1e-10)
            features = features.with_columns([bb_dist.alias(f"bb_distance_{window}")])

        # ATR (Average True Range) normalized
        high_low = features["high"] - features["low"]
        high_close = (features["high"] - features["close"].shift(1)).abs()
        low_close = (features["low"] - features["close"].shift(1)).abs()

        true_range = pl.max_horizontal([high_low, high_close, low_close])

        for window in [14, 28]:
            atr = true_range.rolling_mean(window)
            atr_normalized = atr / (features["close"] + 1e-10)
            features = features.with_columns([atr_normalized.alias(f"atr_normalized_{window}")])

        # MACD
        ema_12 = features["close"].ewm_mean(span=12)
        ema_26 = features["close"].ewm_mean(span=26)
        macd = ema_12 - ema_26
        macd_signal = macd.ewm_mean(span=9)

        features = features.with_columns([
            macd.alias("macd"),
            macd_signal.alias("macd_signal"),
            (macd - macd_signal).alias("macd_histogram")
        ])

        return features

    def _add_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add cyclical time-based features."""
        features = df

        # Extract time components
        hour = features["timestamp"].dt.hour()
        day_of_week = features["timestamp"].dt.weekday()

        # Cyclical encoding
        features = features.with_columns([
            (np.sin(2 * np.pi * hour / 24)).alias("hour_sin"),
            (np.cos(2 * np.pi * hour / 24)).alias("hour_cos"),
            (np.sin(2 * np.pi * day_of_week / 7)).alias("day_of_week_sin"),
            (np.cos(2 * np.pi * day_of_week / 7)).alias("day_of_week_cos")
        ])

        return features

    def _add_lag_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add lagged price and volume features."""
        features = df

        # Price lags
        for lag in [1, 5, 15, 30]:
            features = features.with_columns([
                pl.col("close").shift(lag).alias(f"close_lag_{lag}")
            ])

        # Return lags
        returns = np.log(features["close"] / features["close"].shift(1))
        for lag in [1, 5, 15]:
            features = features.with_columns([
                returns.shift(lag).alias(f"return_lag_{lag}")
            ])

        return features

    def _add_rolling_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add rolling statistical features."""
        features = df

        returns = np.log(features["close"] / features["close"].shift(1))

        for window in [30, 60, 240]:
            # Rolling mean return
            features = features.with_columns([
                returns.rolling_mean(window).alias(f"return_mean_{window}")
            ])

            # Rolling skewness (approximation)
            mean_ret = returns.rolling_mean(window)
            std_ret = returns.rolling_std(window)

            # Simplified skewness
            z_scores = (returns - mean_ret) / (std_ret + 1e-10)
            skew_approx = (z_scores ** 3).rolling_mean(window)

            features = features.with_columns([
                skew_approx.alias(f"return_skew_{window}")
            ])

        return features

    def add_funding_oi_features(
        self,
        df: pl.DataFrame,
        funding_rate: float,
        oi: float,
        oi_prev: float
    ) -> pl.DataFrame:
        """
        Add funding rate and open interest features.

        Args:
            df: Base feature DataFrame
            funding_rate: Current funding rate
            oi: Current open interest
            oi_prev: Previous open interest

        Returns:
            DataFrame with funding/OI features
        """
        # Add as constant columns for the latest row
        df = df.with_columns([
            pl.lit(funding_rate).alias("funding_rate"),
            pl.lit(oi).alias("open_interest"),
            pl.lit((oi - oi_prev) / (oi_prev + 1e-10)).alias("oi_delta_pct")
        ])

        # Funding rate EMA (compute if we have history)
        if "funding_rate" in df.columns:
            df = df.with_columns([
                pl.col("funding_rate").ewm_mean(span=8).alias("funding_rate_ema")
            ])

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of all computed feature names."""
        # This should match all features created in compute_all_features
        base_features = [
            # Orderbook
            "micro_price", "spread_bps", "orderflow_imbalance", "book_depth_ratio",

            # Volume
            "volume_delta", "cumulative_delta",
            "volume_ma_ratio_15", "volume_ma_ratio_30", "volume_ma_ratio_60",
            "buy_sell_ratio_15",

            # Volatility
            "realized_vol_15", "realized_vol_60",
            "parkinson_vol_15", "parkinson_vol_60",
            "garman_klass_vol_15", "garman_klass_vol_60",

            # Technical
            "rsi_14", "rsi_28",
            "bb_distance_20", "bb_distance_40",
            "atr_normalized_14", "atr_normalized_28",
            "macd", "macd_signal", "macd_histogram",

            # Temporal
            "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",

            # Lags
            "close_lag_1", "close_lag_5", "close_lag_15", "close_lag_30",
            "return_lag_1", "return_lag_5", "return_lag_15",

            # Rolling stats
            "return_mean_30", "return_mean_60", "return_mean_240",
            "return_skew_30", "return_skew_60", "return_skew_240",

            # Funding/OI
            "funding_rate", "funding_rate_ema", "open_interest", "oi_delta_pct"
        ]

        return base_features

    def normalize_features(
        self,
        df: pl.DataFrame,
        method: str = "rolling_zscore",
        window: int = 240
    ) -> pl.DataFrame:
        """
        Normalize features using rolling statistics.

        Args:
            df: DataFrame with features
            method: Normalization method ("rolling_zscore", "minmax", "robust")
            window: Rolling window for statistics

        Returns:
            DataFrame with normalized features
        """
        feature_cols = [col for col in df.columns if col not in ["timestamp", "symbol"]]

        if method == "rolling_zscore":
            for col in feature_cols:
                mean = df[col].rolling_mean(window)
                std = df[col].rolling_std(window)
                df = df.with_columns([
                    ((pl.col(col) - mean) / (std + 1e-10)).alias(col)
                ])

        elif method == "minmax":
            for col in feature_cols:
                min_val = df[col].rolling_min(window)
                max_val = df[col].rolling_max(window)
                df = df.with_columns([
                    ((pl.col(col) - min_val) / (max_val - min_val + 1e-10)).alias(col)
                ])

        elif method == "robust":
            for col in feature_cols:
                median = df[col].rolling_median(window)
                # Approximate IQR using quantiles
                q75 = df[col].rolling_quantile(0.75, window_size=window)
                q25 = df[col].rolling_quantile(0.25, window_size=window)
                iqr = q75 - q25
                df = df.with_columns([
                    ((pl.col(col) - median) / (iqr + 1e-10)).alias(col)
                ])

        return df


if __name__ == "__main__":
    # Test feature engine
    import time

    # Generate sample data
    n = 10000
    timestamps = pl.datetime_range(
        start=pl.datetime(2024, 1, 1),
        end=pl.datetime(2024, 1, 1, 0, n),
        interval="1m",
        eager=True
    )

    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": np.random.randn(n).cumsum() + 40000,
        "high": np.random.randn(n).cumsum() + 40050,
        "low": np.random.randn(n).cumsum() + 39950,
        "close": np.random.randn(n).cumsum() + 40000,
        "volume": np.random.rand(n) * 1000000,
        "best_bid": np.random.randn(n).cumsum() + 39995,
        "best_ask": np.random.randn(n).cumsum() + 40005,
        "bid_size": np.random.rand(n) * 100,
        "ask_size": np.random.rand(n) * 100
    })

    engine = FeatureEngine()

    start = time.time()
    df_features = engine.compute_all_features(df)
    elapsed = time.time() - start

    print(f"✓ Computed {len(df_features.columns)} features in {elapsed:.3f}s")
    print(f"  Rows: {len(df_features):,}")
    print(f"  Speed: {len(df_features) / elapsed:.0f} rows/sec")
    print(f"\nFeature columns: {df_features.columns}")
