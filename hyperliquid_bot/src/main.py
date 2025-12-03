"""
Main trading bot orchestrator.

Coordinates:
- Real-time data ingestion
- Feature computation
- Model predictions
- LLM ensemble
- Trade execution
- Risk management
- Monitoring
"""

import asyncio
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from logging.handlers import RotatingFileHandler

import polars as pl
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.data_pipeline.hyperliquid_client import HyperliquidClient
from src.features.feature_engine import FeatureEngine
from src.models.lgbm_trainer import IncrementalLGBMTrainer, RollingRetrainer
from src.llm.ollama_ensemble import LLMEnsemble
from src.execution.order_manager import HyperliquidExecutor
from src.risk.risk_manager import RiskManager
from src.monitoring.alerts import DiscordAlerter, HealthMonitor, MetricsCollector


def setup_logging(config):
    """Configure logging system."""
    log_config = config.logging

    # Create logs directory
    log_file = Path(log_config["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_config["level"]))

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=log_config["max_file_size_mb"] * 1024 * 1024,
        backupCount=log_config["backup_count"]
    )
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(log_config["format"])
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class HyperliquidTradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config_path: str = "./config/config.json"):
        # Load configuration
        self.config = load_config(config_path)

        # Setup logging
        self.logger = setup_logging(self.config)
        self.logger.info("=" * 60)
        self.logger.info("Hyperliquid Trading Bot Starting")
        self.logger.info("=" * 60)

        # Initialize components
        self.data_client = None
        self.feature_engine = None
        self.trainer = None
        self.retrainer = None
        self.llm_ensemble = None
        self.executor = None
        self.risk_manager = None
        self.alerter = None
        self.health_monitor = None
        self.metrics_collector = None

        # State
        self.is_running = False
        self.data_buffer = []
        self.current_position = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all bot components."""
        self.logger.info("Initializing components...")

        # Data pipeline
        self.data_client = HyperliquidClient(
            api_url=self.config.get_api_url(),
            symbols=self.config.hyperliquid["symbols"],
            buffer_size=self.config.data_pipeline["buffer_size"]
        )

        # Feature engine
        self.feature_engine = FeatureEngine(
            lookback_windows=self.config.features["lookback_windows"]
        )

        # LightGBM trainer
        self.trainer = IncrementalLGBMTrainer(
            num_classes=self.config.model["num_classes"],
            class_labels=self.config.model["class_labels"],
            training_params=self.config.model["training_params"],
            model_save_path=self.config.model["model_save_path"]
        )

        # Try to load existing model
        symbol = self.config.hyperliquid["default_symbol"]
        model_path = self.config.get_model_path(symbol)

        if model_path.exists():
            try:
                self.trainer.load_model(model_path)
                self.logger.info(f"✓ Loaded existing model from {model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load model: {e}")
                self.logger.info("Will train new model on startup")

        # Rolling retrainer
        if self.config.model["incremental_learning"]:
            self.retrainer = RollingRetrainer(
                trainer=self.trainer,
                retrain_interval_minutes=self.config.model["retrain_interval_minutes"],
                min_samples=self.config.model["min_samples_for_retrain"],
                max_samples=self.config.model["max_train_samples"]
            )

        # LLM ensemble
        if self.config.llm["enabled"]:
            self.llm_ensemble = LLMEnsemble(
                ollama_host=self.config.llm["ollama_host"],
                main_model_config=self.config.llm["main_model"],
                fast_model_config=self.config.llm["fast_model"],
                ensemble_weight=self.config.llm["ensemble_weight"],
                lgbm_weight=self.config.llm["lgbm_weight"],
                override_threshold=self.config.llm["override_threshold"]
            )

        # Execution engine
        if self.config.execution["enabled"]:
            self.executor = HyperliquidExecutor(
                api_url=self.config.get_api_url(),
                wallet_address=self.config.hyperliquid["wallet_address"],
                private_key="",  # Load from file in production
                is_testnet=self.config.is_testnet()
            )

        # Risk manager
        self.risk_manager = RiskManager(
            max_position_size_usd=self.config.risk["max_position_size_usd"],
            max_leverage=self.config.risk["max_leverage"],
            kelly_fraction=self.config.risk["kelly_fraction"],
            kelly_lookback=self.config.risk["kelly_lookback_trades"],
            stop_loss_pct=self.config.risk["stop_loss_pct"],
            take_profit_pct=self.config.risk["take_profit_pct"],
            min_confidence_to_trade=self.config.risk["min_confidence_to_trade"]
        )

        # Monitoring
        self.alerter = DiscordAlerter(
            webhook_url=self.config.monitoring.get("discord_webhook_url"),
            enabled=self.config.monitoring["enable_discord"],
            alert_on_trade=self.config.monitoring["alert_on_trade"],
            alert_on_stop=self.config.monitoring["alert_on_stop_loss"],
            alert_on_error=self.config.monitoring["alert_on_error"]
        )

        self.health_monitor = HealthMonitor(
            check_interval=self.config.monitoring["health_check_interval_seconds"]
        )

        self.metrics_collector = MetricsCollector()

        self.logger.info("✓ All components initialized")

    async def start(self):
        """Start the trading bot."""
        self.is_running = True
        self.logger.info("🚀 Trading bot started")

        # Send startup alert
        self.alerter.send_alert(
            title="🚀 Bot Started",
            message=f"Hyperliquid bot started in {self.config.environment['mode']} mode",
            alert_type="info",
            fields={
                "Mode": self.config.environment["mode"],
                "Symbols": ", ".join(self.config.hyperliquid["symbols"]),
                "Execution": "Enabled" if self.config.execution["enabled"] else "Disabled"
            }
        )

        # Start rolling retrainer if enabled
        if self.retrainer:
            self.retrainer.start()

        # Fetch initial historical data to bootstrap the bot
        self.logger.info("Fetching initial historical candle data...")
        await self._fetch_initial_data()

        # Start data client in background
        data_task = asyncio.create_task(self.data_client.start())

        # Start main trading loop
        trading_task = asyncio.create_task(self._trading_loop())

        # Start monitoring loop
        monitoring_task = asyncio.create_task(self._monitoring_loop())

        # Wait for all tasks
        try:
            await asyncio.gather(data_task, trading_task, monitoring_task)
        except asyncio.CancelledError:
            self.logger.info("Received shutdown signal")
            await self.shutdown()

    async def _fetch_initial_data(self):
        """Fetch initial historical data to bootstrap the bot."""
        from datetime import datetime, timedelta

        for symbol in self.config.hyperliquid["symbols"]:
            try:
                # Fetch last 6 hours of candles (360 candles)
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=6)

                api_symbol = symbol.split("-")[0]  # BTC-PERP -> BTC

                candles_df = self.data_client.get_historical_candles(
                    symbol=api_symbol,
                    interval="1m",
                    start_time=start_time,
                    end_time=end_time
                )

                if not candles_df.is_empty():
                    # Add to buffer
                    for row in candles_df.iter_rows(named=True):
                        candle_data = {
                            "timestamp": row["timestamp"].timestamp() if hasattr(row["timestamp"], "timestamp") else row["timestamp"],
                            "symbol": symbol,
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"])
                        }
                        self.data_client.candles_buffer.append(candle_data)

                    self.logger.info(f"✓ Loaded {len(candles_df)} historical candles for {symbol}")
                else:
                    self.logger.warning(f"No historical data available for {symbol}")

            except Exception as e:
                self.logger.error(f"Error fetching initial data for {symbol}: {e}")

    async def _trading_loop(self):
        """Main trading loop."""
        self.logger.info("Trading loop started")

        last_prediction_time = 0
        last_macro_analysis_time = 0
        last_orderflow_analysis_time = 0

        while self.is_running:
            try:
                # Get latest candle data
                candles_df = self.data_client.get_buffer_dataframe("candles")

                if len(candles_df) < 300:  # Need minimum history
                    await asyncio.sleep(1)
                    continue

                # Get latest orderbook and trades
                orderbook_df = self.data_client.get_buffer_dataframe("orderbook")
                trades_df = self.data_client.get_buffer_dataframe("trades")

                # Compute features
                symbol = self.config.hyperliquid["default_symbol"]
                df = self._prepare_features(candles_df, orderbook_df, trades_df)

                if df.is_empty():
                    await asyncio.sleep(1)
                    continue

                # Update health
                self.health_monitor.update_metric("last_data_update", time.time())
                self.health_monitor.update_metric("model_loaded", self.trainer.model is not None)

                # Make prediction
                if self.trainer.model is not None:
                    start_time = time.time()

                    # Get latest features
                    feature_names = self.feature_engine.get_feature_names()
                    X = df.select(feature_names).tail(1).to_numpy()

                    if X.shape[0] > 0 and not np.any(np.isnan(X)):
                        preds, probs = self.trainer.predict(X)

                        pred_class = preds[0]
                        lgbm_conf = probs[0][pred_class]

                        prediction_time = (time.time() - start_time) * 1000
                        self.metrics_collector.record_prediction(prediction_time)

                        # LLM ensemble (if enabled)
                        if self.llm_ensemble:
                            # Macro analysis (periodic)
                            if self.llm_ensemble.should_run_macro_analysis():
                                await self._run_macro_analysis(symbol, df)

                            # Order flow analysis (frequent)
                            if self.llm_ensemble.should_run_orderflow_analysis():
                                await self._run_orderflow_analysis(trades_df)

                            # Combine signals
                            final_class, final_conf, details = self.llm_ensemble.combine_signals(
                                probs[0],
                                use_llm=True
                            )
                        else:
                            final_class = pred_class
                            final_conf = lgbm_conf
                            details = {}

                        # Execute trade logic
                        await self._execute_trade_logic(
                            symbol=symbol,
                            prediction_class=final_class,
                            confidence=final_conf,
                            current_price=float(df["close"].tail(1)[0]),
                            details=details
                        )

                # Add data to retraining buffer
                if self.retrainer and len(df) > 0:
                    self.retrainer.add_data(df.tail(100))

                await asyncio.sleep(5)  # Main loop frequency

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}", exc_info=True)
                self.health_monitor.increment_error_count()
                self.alerter.alert_error("Trading Loop Error", str(e))
                await asyncio.sleep(10)

    def _prepare_features(
        self,
        candles_df: pl.DataFrame,
        orderbook_df: pl.DataFrame,
        trades_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Prepare features from raw data."""
        if candles_df.is_empty():
            return pl.DataFrame()

        # ALWAYS add orderbook columns to ensure consistent feature count
        # Use actual data if available, otherwise use default values
        if not orderbook_df.is_empty():
            # Take most recent orderbook snapshot
            latest_book = orderbook_df.tail(1)

            for col in ["best_bid", "best_ask", "bid_size", "ask_size"]:
                if col in latest_book.columns:
                    candles_df = candles_df.with_columns([
                        pl.lit(latest_book[col][0]).alias(col)
                    ])
        else:
            # Add default orderbook columns with NaN values to maintain consistent schema
            # This ensures compute_all_features always sees the same columns
            mid_price = candles_df["close"].tail(1)[0] if len(candles_df) > 0 else 0.0
            candles_df = candles_df.with_columns([
                pl.lit(mid_price).alias("best_bid"),  # Use close price as default
                pl.lit(mid_price).alias("best_ask"),  # Use close price as default
                pl.lit(0.0).alias("bid_size"),  # Zero size when no orderbook
                pl.lit(0.0).alias("ask_size")   # Zero size when no orderbook
            ])

        # Compute all features
        df = self.feature_engine.compute_all_features(candles_df)

        # Add funding/OI if available
        symbol = self.config.hyperliquid["default_symbol"]
        funding_rate = self.data_client.current_funding.get(symbol, 0.0)
        oi = self.data_client.current_oi.get(symbol, 0.0)

        # Simple OI delta (compare to previous)
        oi_buffer = list(self.data_client.oi_buffer)
        oi_prev = oi_buffer[-2]["open_interest"] if len(oi_buffer) > 1 else oi

        df = self.feature_engine.add_funding_oi_features(df, funding_rate, oi, oi_prev)

        # Add labels for training (based on future price movement)
        df = self._add_labels(df)

        # Normalize features
        df = self.feature_engine.normalize_features(
            df,
            method=self.config.features["normalization"]
        )

        return df

    def _add_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add target labels based on forward price movement.

        Labels:
        - 0: Short (price decreases > threshold)
        - 1: Flat (price stays within threshold)
        - 2: Long (price increases > threshold)
        """
        if df.is_empty() or len(df) < 2:
            return df

        # Get labeling params from config
        forward_bars = self.config.features.get("labeling", {}).get("forward_bars", 60)
        profit_threshold = self.config.features.get("labeling", {}).get("profit_threshold", 0.005)

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

        # Drop forward_return column (only needed for label generation)
        df = df.drop("forward_return")

        return df

    async def _run_macro_analysis(self, symbol: str, df: pl.DataFrame):
        """Run periodic macro market analysis."""
        try:
            # Calculate 6-hour metrics
            recent = df.tail(360)  # 6 hours of 1-min candles

            if len(recent) < 100:
                return

            price_change = ((recent["close"][-1] - recent["close"][0]) / recent["close"][0]) * 100
            funding_rate = self.data_client.current_funding.get(symbol, 0.0)

            oi_buffer = list(self.data_client.oi_buffer)
            if len(oi_buffer) > 2:
                oi_change = ((oi_buffer[-1]["open_interest"] - oi_buffer[-360]["open_interest"]) /
                            oi_buffer[-360]["open_interest"]) * 100
            else:
                oi_change = 0.0

            # Analyze large trades
            trades_buffer = list(self.data_client.trades_buffer)
            large_trades = [t for t in trades_buffer[-100:] if t["size"] > 1.0]
            whale_flow = f"{len(large_trades)} large trades, avg size: {np.mean([t['size'] for t in large_trades]) if large_trades else 0:.2f}"

            volume_profile = "balanced"  # Simplified

            result = self.llm_ensemble.analyze_macro(
                symbol=symbol,
                price_change_pct=price_change,
                funding_rate=funding_rate,
                oi_change_pct=oi_change,
                whale_flow=whale_flow,
                volume_profile=volume_profile
            )

            self.metrics_collector.record_llm_inference(1000)  # Approximate

        except Exception as e:
            self.logger.error(f"Macro analysis error: {e}")

    async def _run_orderflow_analysis(self, trades_df: pl.DataFrame):
        """Run fast order flow analysis."""
        try:
            if trades_df.is_empty():
                return

            recent_trades = trades_df.tail(60)

            # Calculate order flow metrics
            buys = recent_trades.filter(pl.col("side") == "B")
            sells = recent_trades.filter(pl.col("side") == "A")

            buy_volume = buys["size"].sum() if len(buys) > 0 else 0
            sell_volume = sells["size"].sum() if len(sells) > 0 else 0

            cum_delta = buy_volume - sell_volume
            buy_sell_ratio = buy_volume / (sell_volume + 1e-10)

            # Get orderbook imbalance
            symbol = self.config.hyperliquid["default_symbol"]
            if symbol in self.data_client.current_orderbook:
                book = self.data_client.current_orderbook[symbol]
                imbalance = (book["bid_size"] - book["ask_size"]) / (book["bid_size"] + book["ask_size"] + 1e-10)
            else:
                imbalance = 0.0

            large_trades = [t for t in recent_trades.to_dicts() if t["size"] > 0.5]
            large_trades_desc = f"{len(large_trades)} trades > 0.5 BTC"

            result = self.llm_ensemble.analyze_orderflow(
                cum_delta=cum_delta,
                buy_sell_ratio=buy_sell_ratio,
                imbalance=imbalance,
                large_trades=large_trades_desc
            )

        except Exception as e:
            self.logger.error(f"Order flow analysis error: {e}")

    async def _execute_trade_logic(
        self,
        symbol: str,
        prediction_class: int,
        confidence: float,
        current_price: float,
        details: Dict
    ):
        """Execute trading logic based on prediction."""
        class_labels = self.config.model["class_labels"]
        predicted_action = class_labels[prediction_class]

        # Check if can trade
        can_trade, reason = self.risk_manager.check_can_trade(symbol, confidence)

        if not can_trade:
            self.logger.debug(f"Cannot trade: {reason}")
            return

        # Check current position
        if self.executor and self.config.execution["enabled"]:
            position = self.executor.get_position(symbol)
        else:
            position = None

        # Trading logic
        if position is None or abs(position.size) < 1e-8:
            # No position - consider entry
            if predicted_action in ["long", "short"]:
                # Calculate position size
                size = self.risk_manager.calculate_kelly_size(
                    confidence=confidence,
                    current_price=current_price,
                    account_balance=self.risk_manager.account_balance
                )

                if size > 0:
                    self.logger.info(f"Signal: {predicted_action.upper()} @ ${current_price:.2f} (conf: {confidence:.2f})")

                    if self.executor and self.config.execution["enabled"]:
                        # Execute order
                        side = "buy" if predicted_action == "long" else "sell"

                        order = self.executor.place_order(
                            symbol=symbol,
                            side=side,
                            size=size,
                            price=current_price,
                            time_in_force=self.config.execution["time_in_force"]
                        )

                        if order:
                            # Set stops
                            self.risk_manager.on_position_open(symbol, current_price, predicted_action)

                            # Alert
                            self.alerter.alert_trade_opened(
                                symbol=symbol,
                                side=predicted_action,
                                size=size,
                                price=current_price,
                                confidence=confidence
                            )

                            self.metrics_collector.record_trade()

        else:
            # Have position - check stops and exit signals
            position_side = "long" if position.size > 0 else "short"

            # Update trailing stop
            self.risk_manager.update_trailing_stop(symbol, current_price, position.entry_price, position_side)

            # Check stops
            stop_triggered, stop_type = self.risk_manager.check_stops(symbol, current_price, position_side)

            if stop_triggered:
                self.logger.info(f"Stop triggered: {stop_type}")

                if self.executor and self.config.execution["enabled"]:
                    # Close position
                    self.executor.close_position(symbol, current_price)

                    # Calculate PnL
                    if position_side == "long":
                        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    else:
                        pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100

                    pnl = (pnl_pct / 100) * (position.size * position.entry_price)

                    # Update risk manager
                    self.risk_manager.on_position_close(
                        symbol, position.entry_price, current_price,
                        position_side, abs(position.size), stop_type
                    )

                    # Alert
                    self.alerter.alert_trade_closed(
                        symbol=symbol,
                        side=position_side,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=stop_type
                    )

    async def _monitoring_loop(self):
        """Monitoring and health check loop."""
        while self.is_running:
            try:
                # Health check
                is_healthy = self.health_monitor.check_health()

                if not is_healthy:
                    status = self.health_monitor.get_status()
                    self.logger.warning(f"Health check failed: {status}")

                # Log metrics periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    metrics = self.metrics_collector.get_metrics()
                    risk_metrics = self.risk_manager.get_risk_metrics()

                    self.logger.info("=" * 60)
                    self.logger.info("System Metrics:")
                    self.logger.info(f"  Predictions: {metrics['predictions_made']}")
                    self.logger.info(f"  Trades: {metrics['trades_executed']}")
                    self.logger.info(f"  Model retrains: {metrics['model_retrains']}")
                    self.logger.info(f"  Account balance: ${risk_metrics.get('account_balance', 0):.2f}")
                    self.logger.info(f"  Win rate: {risk_metrics.get('win_rate', 0):.1f}%")
                    self.logger.info("=" * 60)

                await asyncio.sleep(self.config.monitoring["health_check_interval_seconds"])

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    async def shutdown(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down trading bot...")

        self.is_running = False

        # Stop data client
        if self.data_client:
            await self.data_client.stop()

        # Stop retrainer
        if self.retrainer:
            self.retrainer.stop()

        # Save model
        if self.trainer and self.trainer.model:
            symbol = self.config.hyperliquid["default_symbol"]
            self.trainer.save_model(symbol=symbol)

        # Send shutdown alert
        self.alerter.send_alert(
            title="🛑 Bot Stopped",
            message="Hyperliquid bot has been stopped",
            alert_type="warning"
        )

        self.logger.info("✓ Shutdown complete")


async def main():
    """Main entry point."""
    bot = None

    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        if bot:
            asyncio.create_task(bot.shutdown())

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start bot
    bot = HyperliquidTradingBot()
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
