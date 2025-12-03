"""
Run walk-forward backtest on historical data.

Usage:
    python -m src.scripts.run_backtest --start 2024-01-01 --end 2024-12-01
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

import polars as pl

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.features.feature_engine import FeatureEngine
from src.models.lgbm_trainer import IncrementalLGBMTrainer
from src.risk.risk_manager import RiskManager
from src.backtest.backtester import WalkForwardBacktester
from src.data_pipeline.hyperliquid_client import HyperliquidClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_historical_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    api_url: str
) -> pl.DataFrame:
    """Fetch historical candle data."""
    logger.info(f"Fetching historical data for {symbol}...")
    logger.info(f"  Period: {start_date} to {end_date}")

    client = HyperliquidClient(api_url=api_url, symbols=[symbol])

    df = client.get_historical_candles(
        symbol=symbol,
        interval="1m",
        start_time=start_date,
        end_time=end_date
    )

    logger.info(f"✓ Fetched {len(df):,} candles")
    return df


def run_backtest(
    start_date: str,
    end_date: str,
    symbol: str = "BTC-PERP",
    config_path: str = "./config/config.json"
):
    """Run walk-forward backtest."""
    logger.info("=" * 60)
    logger.info("Hyperliquid Walk-Forward Backtest")
    logger.info("=" * 60)

    # Load config
    config = load_config(config_path)

    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Fetch historical data
    df = fetch_historical_data(
        symbol=symbol,
        start_date=start_dt,
        end_date=end_dt,
        api_url=config.get_api_url()
    )

    if df.is_empty():
        logger.error("No historical data available")
        return

    # Initialize components
    logger.info("\nInitializing backtest components...")

    feature_engine = FeatureEngine(
        lookback_windows=config.features["lookback_windows"]
    )

    trainer = IncrementalLGBMTrainer(
        num_classes=config.model["num_classes"],
        class_labels=config.model["class_labels"],
        training_params=config.model["training_params"]
    )

    risk_manager = RiskManager(
        max_position_size_usd=config.backtest["initial_capital"] * 0.5,
        kelly_fraction=config.risk["kelly_fraction"],
        stop_loss_pct=config.risk["stop_loss_pct"],
        take_profit_pct=config.risk["take_profit_pct"]
    )

    backtester = WalkForwardBacktester(
        feature_engine=feature_engine,
        trainer=trainer,
        risk_manager=risk_manager,
        initial_capital=config.backtest["initial_capital"],
        commission_bps=config.backtest["commission_bps"],
        slippage_bps=config.backtest["slippage_bps"]
    )

    # Compute features
    logger.info("Computing features...")

    # Add dummy orderbook data (simplified for backtest)
    df = df.with_columns([
        pl.col("close").alias("best_bid"),
        pl.col("close") * 1.0001.alias("best_ask"),
        pl.lit(1.0).alias("bid_size"),
        pl.lit(1.0).alias("ask_size")
    ])

    df_features = feature_engine.compute_all_features(df)

    # Add dummy funding/OI
    df_features = feature_engine.add_funding_oi_features(
        df_features,
        funding_rate=0.01,
        oi=1000000,
        oi_prev=1000000
    )

    # Normalize
    df_features = feature_engine.normalize_features(df_features)

    # Create labels
    logger.info("Creating labels...")
    df_labeled = backtester.prepare_labels(
        df_features,
        forward_bars=60,
        profit_threshold=0.005
    )

    logger.info(f"Dataset prepared: {len(df_labeled):,} samples")

    # Run backtest
    logger.info("\nRunning walk-forward backtest...")

    results = backtester.run_walk_forward(
        df=df_labeled,
        n_periods=config.backtest["walk_forward_periods"],
        retrain_every_days=config.backtest["retrain_frequency_days"]
    )

    # Save results
    output_dir = Path("./data/backtests")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save equity curve
    equity_df = backtester.get_equity_curve()
    equity_path = output_dir / f"equity_curve_{timestamp}.csv"
    equity_df.to_csv(equity_path, index=False)
    logger.info(f"\n✓ Equity curve saved to: {equity_path}")

    # Save trades
    trades_df = backtester.get_trades_df()
    trades_path = output_dir / f"trades_{timestamp}.csv"
    trades_df.to_csv(trades_path, index=False)
    logger.info(f"✓ Trades saved to: {trades_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)

    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run walk-forward backtest")

    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-01",
        help="End date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC-PERP",
        help="Trading symbol"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./config/config.json",
        help="Path to config file"
    )

    args = parser.parse_args()

    try:
        run_backtest(
            start_date=args.start,
            end_date=args.end,
            symbol=args.symbol,
            config_path=args.config
        )
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)
