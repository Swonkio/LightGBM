"""
Walk-forward backtesting system for strategy validation.

Features:
- Walk-forward cross-validation
- Periodic model retraining
- Transaction costs (commission + slippage)
- Funding rate simulation
- Detailed performance metrics
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import polars as pl
import pandas as pd

from ..features.feature_engine import FeatureEngine
from ..models.lgbm_trainer import IncrementalLGBMTrainer
from ..risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Backtest trade record."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    commission: float
    funding: float
    confidence: float


class WalkForwardBacktester:
    """
    Walk-forward backtesting with realistic execution simulation.

    Methodology:
    1. Split data into N periods
    2. For each period:
       - Train on past data
       - Test on forward period
       - Record performance
    3. Aggregate results across all periods
    """

    def __init__(
        self,
        feature_engine: FeatureEngine,
        trainer: IncrementalLGBMTrainer,
        risk_manager: RiskManager,
        initial_capital: float = 10000,
        commission_bps: float = 2.0,
        slippage_bps: float = 1.0,
        funding_rate_hourly: float = 0.01
    ):
        self.feature_engine = feature_engine
        self.trainer = trainer
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.funding_rate_hourly = funding_rate_hourly

        # Results
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

    def prepare_labels(
        self,
        df: pl.DataFrame,
        forward_bars: int = 60,
        profit_threshold: float = 0.005
    ) -> pl.DataFrame:
        """
        Create trading labels based on forward returns.

        Labels:
        - 0: Short (price decreases > threshold)
        - 1: Flat (price stays within threshold)
        - 2: Long (price increases > threshold)

        Args:
            df: DataFrame with OHLCV data
            forward_bars: Bars to look ahead
            profit_threshold: Threshold for directional move

        Returns:
            DataFrame with 'target' column
        """
        # Calculate forward returns
        df = df.with_columns([
            (pl.col("close").shift(-forward_bars) / pl.col("close") - 1).alias("forward_return")
        ])

        # Create labels
        df = df.with_columns([
            pl.when(pl.col("forward_return") > profit_threshold)
            .then(2)  # Long
            .when(pl.col("forward_return") < -profit_threshold)
            .then(0)  # Short
            .otherwise(1)  # Flat
            .alias("target")
        ])

        # Drop rows without forward data
        df = df.filter(pl.col("target").is_not_null())

        return df

    def run_walk_forward(
        self,
        df: pl.DataFrame,
        n_periods: int = 12,
        retrain_every_days: int = 7
    ) -> Dict:
        """
        Run walk-forward backtest.

        Args:
            df: Historical OHLCV data with features
            n_periods: Number of walk-forward periods
            retrain_every_days: Retrain frequency in days

        Returns:
            Performance metrics dictionary
        """
        logger.info(f"Starting walk-forward backtest with {n_periods} periods")

        # Split data into periods
        total_rows = len(df)
        period_size = total_rows // n_periods

        # Initial training period (use first 2 periods)
        train_end = period_size * 2
        train_df = df[:train_end]

        logger.info(f"Initial training on {len(train_df):,} rows")

        # Train initial model
        train_data, val_data, feature_names = self.trainer.prepare_training_data(train_df)
        model = self.trainer.train_from_scratch(train_data, val_data, num_boost_round=300)
        self.trainer.swap_model(model)

        # Track equity
        current_equity = self.initial_capital
        self.equity_curve = [(df["timestamp"][0], current_equity)]

        # Simulate trading on each forward period
        for period_idx in range(2, n_periods):
            period_start = period_size * period_idx
            period_end = min(period_size * (period_idx + 1), total_rows)

            test_df = df[period_start:period_end]

            logger.info(f"\nPeriod {period_idx}/{n_periods}: Testing on {len(test_df):,} rows")

            # Simulate trading
            current_equity = self._simulate_period(test_df, current_equity, feature_names)

            # Retrain model if needed
            if period_idx % (retrain_every_days // 7) == 0:
                # Use expanding window (all past data)
                retrain_df = df[:period_end]
                logger.info(f"Retraining on {len(retrain_df):,} rows")

                train_data, val_data, _ = self.trainer.prepare_training_data(retrain_df)
                model = self.trainer.train_from_scratch(train_data, val_data, num_boost_round=200)
                self.trainer.swap_model(model)

        # Calculate final metrics
        metrics = self._calculate_metrics()

        logger.info("\n=== Backtest Results ===")
        logger.info(f"Total trades: {metrics['total_trades']}")
        logger.info(f"Win rate: {metrics['win_rate']:.2f}%")
        logger.info(f"Total return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {metrics['max_drawdown_pct']:.2f}%")

        return metrics

    def _simulate_period(
        self,
        df: pl.DataFrame,
        starting_equity: float,
        feature_names: List[str]
    ) -> float:
        """
        Simulate trading on a single period.

        Args:
            df: Period data
            starting_equity: Starting equity for period
            feature_names: Feature column names

        Returns:
            Ending equity
        """
        current_equity = starting_equity
        position = None  # (entry_price, entry_time, side, size, confidence)

        for i in range(len(df)):
            row = df[i]
            current_time = row["timestamp"]
            current_price = row["close"]

            # Check stops if in position
            if position is not None:
                entry_price, entry_time, side, size, confidence = position

                # Check stop loss / take profit
                stop_triggered, stop_type = self.risk_manager.check_stops(
                    "BTC-PERP", current_price, side
                )

                if stop_triggered:
                    # Close position
                    current_equity = self._close_position(
                        position, current_price, current_time, current_equity, stop_type
                    )
                    position = None
                    continue

            # Get features for prediction
            X = df[i:i+1].select(feature_names).to_numpy()

            if X.shape[0] == 0 or np.any(np.isnan(X)):
                continue

            # Predict
            try:
                preds, probs = self.trainer.predict(X)
                pred_class = preds[0]
                confidence = probs[0][pred_class]

                # Trading logic
                if position is None:
                    # Check if should enter
                    can_trade, reason = self.risk_manager.check_can_trade("BTC-PERP", confidence)

                    if can_trade and pred_class in [0, 2]:  # Short or Long
                        side = "long" if pred_class == 2 else "short"

                        # Calculate position size
                        size = self.risk_manager.calculate_kelly_size(
                            confidence, current_price, current_equity
                        )

                        if size > 0:
                            # Apply slippage
                            if side == "long":
                                entry_price = current_price * (1 + self.slippage_bps / 10000)
                            else:
                                entry_price = current_price * (1 - self.slippage_bps / 10000)

                            # Calculate commission
                            commission = (size * entry_price) * (self.commission_bps / 10000)
                            current_equity -= commission

                            # Enter position
                            position = (entry_price, current_time, side, size, confidence)

                            # Set stops
                            self.risk_manager.on_position_open("BTC-PERP", entry_price, side)

                else:
                    # Update trailing stop
                    entry_price, entry_time, side, size, confidence = position
                    self.risk_manager.update_trailing_stop("BTC-PERP", current_price, entry_price, side)

                    # Check if should exit (signal reversal or flat)
                    if (side == "long" and pred_class == 0) or (side == "short" and pred_class == 2):
                        current_equity = self._close_position(
                            position, current_price, current_time, current_equity, "signal_reversal"
                        )
                        position = None

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                continue

        # Close any remaining position at end of period
        if position is not None:
            current_equity = self._close_position(
                position, df[-1]["close"], df[-1]["timestamp"], current_equity, "period_end"
            )

        return current_equity

    def _close_position(
        self,
        position: Tuple,
        exit_price: float,
        exit_time: datetime,
        current_equity: float,
        exit_reason: str
    ) -> float:
        """Close a position and record trade."""
        entry_price, entry_time, side, size, confidence = position

        # Apply slippage
        if side == "long":
            exit_price_adj = exit_price * (1 - self.slippage_bps / 10000)
        else:
            exit_price_adj = exit_price * (1 + self.slippage_bps / 10000)

        # Calculate PnL
        if side == "long":
            pnl_pct = (exit_price_adj - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price_adj) / entry_price

        pnl = pnl_pct * (size * entry_price)

        # Commission
        commission = (size * exit_price_adj) * (self.commission_bps / 10000)

        # Funding (simplified: proportional to hold time)
        hold_hours = (exit_time - entry_time).total_seconds() / 3600
        funding = (size * entry_price) * (self.funding_rate_hourly / 100) * hold_hours

        # Net PnL
        net_pnl = pnl - commission - funding

        # Update equity
        current_equity += net_pnl

        # Record trade
        trade = BacktestTrade(
            entry_time=entry_time,
            exit_time=exit_time,
            symbol="BTC-PERP",
            side=side,
            size=size,
            entry_price=entry_price,
            exit_price=exit_price_adj,
            pnl=net_pnl,
            pnl_pct=pnl_pct * 100,
            commission=commission,
            funding=funding,
            confidence=confidence
        )

        self.trades.append(trade)
        self.equity_curve.append((exit_time, current_equity))

        # Update risk manager
        self.risk_manager.on_position_close(
            "BTC-PERP", entry_price, exit_price_adj, side, size, exit_reason
        )

        return current_equity

    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics."""
        if not self.trades:
            return {}

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]

        total_pnl = sum(t.pnl for t in self.trades)
        total_return_pct = (total_pnl / self.initial_capital) * 100

        win_rate = (len(wins) / len(self.trades)) * 100 if self.trades else 0

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0

        total_wins = sum(t.pnl for t in wins)
        total_losses = sum(abs(t.pnl) for t in losses)
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Calculate Sharpe ratio
        returns = [t.pnl / self.initial_capital for t in self.trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if returns else 0

        # Max drawdown
        equity_values = [eq for _, eq in self.equity_curve]
        peak = self.initial_capital
        max_dd_pct = 0

        for eq in equity_values:
            if eq > peak:
                peak = eq
            dd_pct = ((peak - eq) / peak) * 100
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "total_return_pct": total_return_pct,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd_pct,
            "final_equity": equity_values[-1] if equity_values else self.initial_capital
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([vars(t) for t in self.trades])


if __name__ == "__main__":
    # This would be run with real historical data
    logger.info("Backtest module ready")
