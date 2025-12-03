"""
Monitoring and alerting system with Discord notifications.

Features:
- Discord webhook alerts
- Health checks
- Performance metrics tracking
- Error monitoring
"""

import time
from typing import Dict, Optional
from datetime import datetime
import logging

import requests

logger = logging.getLogger(__name__)


class DiscordAlerter:
    """Send trading alerts to Discord via webhook."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        enabled: bool = True,
        alert_on_trade: bool = True,
        alert_on_stop: bool = True,
        alert_on_error: bool = True
    ):
        self.webhook_url = webhook_url
        self.enabled = enabled and webhook_url is not None
        self.alert_on_trade = alert_on_trade
        self.alert_on_stop = alert_on_stop
        self.alert_on_error = alert_on_error

        # Rate limiting
        self.last_alert_time = {}
        self.min_alert_interval = 10  # Minimum seconds between same alert type

    def send_alert(
        self,
        title: str,
        message: str,
        alert_type: str = "info",
        fields: Optional[Dict] = None
    ):
        """
        Send alert to Discord.

        Args:
            title: Alert title
            message: Alert message
            alert_type: "info", "trade", "stop", "error", "warning"
            fields: Additional fields to display
        """
        if not self.enabled:
            return

        # Rate limiting
        if alert_type in self.last_alert_time:
            if time.time() - self.last_alert_time[alert_type] < self.min_alert_interval:
                logger.debug(f"Rate limited: {alert_type}")
                return

        # Color coding
        color_map = {
            "info": 0x3498db,      # Blue
            "trade": 0x2ecc71,     # Green
            "stop": 0xe74c3c,      # Red
            "error": 0xe74c3c,     # Red
            "warning": 0xf39c12    # Orange
        }

        color = color_map.get(alert_type, 0x95a5a6)

        # Build embed
        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Hyperliquid Trading Bot"}
        }

        # Add fields
        if fields:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in fields.items()
            ]

        payload = {"embeds": [embed]}

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            self.last_alert_time[alert_type] = time.time()
            logger.debug(f"Discord alert sent: {title}")

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    def alert_trade_opened(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        confidence: float
    ):
        """Alert when position is opened."""
        if not self.alert_on_trade:
            return

        self.send_alert(
            title=f"🟢 Position Opened: {symbol}",
            message=f"Entered {side.upper()} position",
            alert_type="trade",
            fields={
                "Side": side.upper(),
                "Size": f"{size:.4f}",
                "Price": f"${price:.2f}",
                "Confidence": f"{confidence:.2%}",
                "Position Value": f"${size * price:.2f}"
            }
        )

    def alert_trade_closed(
        self,
        symbol: str,
        side: str,
        pnl: float,
        pnl_pct: float,
        exit_reason: str
    ):
        """Alert when position is closed."""
        if not self.alert_on_trade:
            return

        emoji = "🟢" if pnl > 0 else "🔴"
        title = f"{emoji} Position Closed: {symbol}"

        self.send_alert(
            title=title,
            message=f"Closed {side.upper()} position",
            alert_type="trade" if pnl > 0 else "stop",
            fields={
                "Side": side.upper(),
                "PnL": f"${pnl:.2f}",
                "PnL %": f"{pnl_pct:.2f}%",
                "Reason": exit_reason
            }
        )

    def alert_stop_loss(
        self,
        symbol: str,
        side: str,
        pnl: float,
        stop_type: str
    ):
        """Alert when stop loss is triggered."""
        if not self.alert_on_stop:
            return

        self.send_alert(
            title=f"🛑 Stop Loss Triggered: {symbol}",
            message=f"{stop_type.replace('_', ' ').title()} hit",
            alert_type="stop",
            fields={
                "Side": side.upper(),
                "PnL": f"${pnl:.2f}",
                "Stop Type": stop_type.replace("_", " ").title()
            }
        )

    def alert_error(self, error_type: str, error_msg: str):
        """Alert on system errors."""
        if not self.alert_on_error:
            return

        self.send_alert(
            title=f"⚠️ Error: {error_type}",
            message=error_msg,
            alert_type="error"
        )

    def alert_daily_summary(self, metrics: Dict):
        """Send daily performance summary."""
        self.send_alert(
            title="📊 Daily Summary",
            message=f"Trading performance for {datetime.now().strftime('%Y-%m-%d')}",
            alert_type="info",
            fields={
                "Total Trades": metrics.get("total_trades", 0),
                "Win Rate": f"{metrics.get('win_rate', 0):.1f}%",
                "Daily PnL": f"${metrics.get('daily_pnl', 0):.2f}",
                "Account Balance": f"${metrics.get('account_balance', 0):.2f}",
                "Drawdown": f"{metrics.get('current_drawdown_pct', 0):.2f}%"
            }
        )


class HealthMonitor:
    """System health monitoring."""

    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.last_check_time = 0

        # Health metrics
        self.metrics = {
            "data_pipeline_healthy": False,
            "model_loaded": False,
            "llm_available": False,
            "last_data_update": 0,
            "last_model_update": 0,
            "error_count": 0
        }

    def update_metric(self, key: str, value):
        """Update a health metric."""
        self.metrics[key] = value

    def increment_error_count(self):
        """Increment error counter."""
        self.metrics["error_count"] += 1

    def reset_error_count(self):
        """Reset error counter."""
        self.metrics["error_count"] = 0

    def check_health(self) -> bool:
        """
        Perform health check.

        Returns:
            True if system is healthy
        """
        now = time.time()

        if now - self.last_check_time < self.check_interval:
            return True

        self.last_check_time = now

        # Check data freshness
        data_age = now - self.metrics.get("last_data_update", 0)
        data_healthy = data_age < 300  # Data should be < 5 min old

        # Check model availability
        model_healthy = self.metrics.get("model_loaded", False)

        # Check error rate
        error_healthy = self.metrics.get("error_count", 0) < 10

        is_healthy = data_healthy and model_healthy and error_healthy

        if not is_healthy:
            logger.warning("Health check failed:")
            if not data_healthy:
                logger.warning(f"  Data stale: {data_age:.0f}s old")
            if not model_healthy:
                logger.warning("  Model not loaded")
            if not error_healthy:
                logger.warning(f"  Error count: {self.metrics['error_count']}")

        return is_healthy

    def get_status(self) -> Dict:
        """Get current system status."""
        return {
            **self.metrics,
            "is_healthy": self.check_health()
        }


class MetricsCollector:
    """Collect and aggregate performance metrics."""

    def __init__(self):
        self.metrics = {
            "predictions_made": 0,
            "trades_executed": 0,
            "model_retrains": 0,
            "llm_inferences": 0,
            "data_points_processed": 0,
            "avg_prediction_time_ms": 0,
            "avg_llm_time_ms": 0
        }

        self.timings = []

    def record_prediction(self, duration_ms: float):
        """Record prediction timing."""
        self.metrics["predictions_made"] += 1
        self.timings.append(duration_ms)

        if len(self.timings) > 1000:
            self.timings = self.timings[-1000:]

        self.metrics["avg_prediction_time_ms"] = sum(self.timings) / len(self.timings)

    def record_trade(self):
        """Record trade execution."""
        self.metrics["trades_executed"] += 1

    def record_retrain(self):
        """Record model retrain."""
        self.metrics["model_retrains"] += 1

    def record_llm_inference(self, duration_ms: float):
        """Record LLM inference."""
        self.metrics["llm_inferences"] += 1
        self.metrics["avg_llm_time_ms"] = duration_ms

    def record_data_points(self, count: int):
        """Record data points processed."""
        self.metrics["data_points_processed"] += count

    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return self.metrics.copy()


if __name__ == "__main__":
    # Test Discord alerts
    alerter = DiscordAlerter(enabled=False)  # Set webhook URL to test

    alerter.alert_trade_opened(
        symbol="BTC-PERP",
        side="long",
        size=0.1,
        price=40000,
        confidence=0.75
    )

    print("✓ Alerter initialized")

    # Test health monitor
    health = HealthMonitor()
    health.update_metric("model_loaded", True)
    health.update_metric("last_data_update", time.time())

    status = health.get_status()
    print(f"✓ Health status: {status}")

    # Test metrics collector
    metrics = MetricsCollector()
    metrics.record_prediction(5.2)
    metrics.record_trade()

    print(f"✓ Metrics: {metrics.get_metrics()}")
