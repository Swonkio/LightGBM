"""
Risk management system with Kelly criterion position sizing.

Features:
- Kelly fraction position sizing based on edge
- Dynamic position limits
- Stop loss and take profit management
- Drawdown monitoring
- Daily loss limits
"""

import time
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade record."""
    timestamp: float
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float


class RiskManager:
    """
    Comprehensive risk management system.

    Implements:
    - Kelly criterion position sizing
    - Stop loss / take profit
    - Trailing stops
    - Drawdown limits
    - Daily loss limits
    """

    def __init__(
        self,
        max_position_size_usd: float = 10000,
        max_leverage: float = 8.0,
        kelly_fraction: float = 0.25,
        kelly_lookback: int = 100,
        min_edge_threshold: float = 0.02,
        max_drawdown_pct: float = 15.0,
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
        trailing_stop_pct: float = 1.5,
        daily_loss_limit_pct: float = 5.0,
        min_confidence_to_trade: float = 0.6
    ):
        self.max_position_size_usd = max_position_size_usd
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
        self.kelly_lookback = kelly_lookback
        self.min_edge_threshold = min_edge_threshold
        self.max_drawdown_pct = max_drawdown_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.min_confidence_to_trade = min_confidence_to_trade

        # Trade history
        self.trade_history: deque = deque(maxlen=kelly_lookback)

        # Active stops
        self.stop_losses: Dict[str, float] = {}
        self.take_profits: Dict[str, float] = {}
        self.trailing_stops: Dict[str, float] = {}
        self.highest_prices: Dict[str, float] = {}

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_start_time = time.time()

        # Cooldown tracking (after stop loss)
        self.cooldown_until: Dict[str, float] = {}

        # Account state
        self.account_balance = 10000.0  # Default starting balance
        self.peak_balance = 10000.0
        self.current_drawdown_pct = 0.0

    def calculate_kelly_size(
        self,
        confidence: float,
        current_price: float,
        account_balance: float
    ) -> float:
        """
        Calculate position size using Kelly criterion.

        Kelly formula: f = (p * b - q) / b
        where:
        - p = win probability (from model confidence)
        - q = 1 - p (loss probability)
        - b = win/loss ratio (from historical trades)

        Args:
            confidence: Model confidence (0-1)
            current_price: Current market price
            account_balance: Available capital

        Returns:
            Position size in base currency (e.g., BTC)
        """
        if not self.trade_history:
            # No history: use conservative fixed fraction
            kelly_fraction = self.kelly_fraction * confidence
            position_usd = account_balance * kelly_fraction
            return position_usd / current_price

        # Calculate win/loss ratio from history
        wins = [t for t in self.trade_history if t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl < 0]

        if not wins or not losses:
            # Incomplete data: use conservative sizing
            kelly_fraction = self.kelly_fraction * confidence
            position_usd = account_balance * kelly_fraction
            return position_usd / current_price

        avg_win = np.mean([t.pnl_pct for t in wins])
        avg_loss = abs(np.mean([t.pnl_pct for t in losses]))

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Kelly formula
        p = confidence
        q = 1 - p
        b = win_loss_ratio

        kelly_f = (p * b - q) / b

        # Apply Kelly fraction (fractional Kelly for safety)
        kelly_f = max(0, min(kelly_f, 1.0))  # Clamp to [0, 1]
        adjusted_kelly = kelly_f * self.kelly_fraction

        # Calculate position size
        position_usd = account_balance * adjusted_kelly

        # Apply max position limit
        position_usd = min(position_usd, self.max_position_size_usd)

        # Convert to base currency
        position_size = position_usd / current_price

        logger.debug(f"Kelly sizing: f={adjusted_kelly:.3f}, position=${position_usd:.2f}")
        return position_size

    def check_can_trade(
        self,
        symbol: str,
        confidence: float
    ) -> Tuple[bool, str]:
        """
        Check if a trade is allowed based on risk rules.

        Args:
            symbol: Trading symbol
            confidence: Model confidence

        Returns:
            (allowed, reason)
        """
        # 1. Check confidence threshold
        if confidence < self.min_confidence_to_trade:
            return False, f"Confidence too low: {confidence:.2f} < {self.min_confidence_to_trade}"

        # 2. Check drawdown limit
        if self.current_drawdown_pct >= self.max_drawdown_pct:
            return False, f"Max drawdown reached: {self.current_drawdown_pct:.2f}%"

        # 3. Check daily loss limit
        daily_loss_pct = (self.daily_pnl / self.account_balance) * 100
        if daily_loss_pct <= -self.daily_loss_limit_pct:
            return False, f"Daily loss limit reached: {daily_loss_pct:.2f}%"

        # 4. Check cooldown
        if symbol in self.cooldown_until:
            if time.time() < self.cooldown_until[symbol]:
                remaining = (self.cooldown_until[symbol] - time.time()) / 60
                return False, f"Cooldown active for {remaining:.1f} minutes"

        return True, "OK"

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str
    ) -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price
            side: "long" or "short"

        Returns:
            Stop loss price
        """
        if side == "long":
            return entry_price * (1 - self.stop_loss_pct / 100)
        else:  # short
            return entry_price * (1 + self.stop_loss_pct / 100)

    def calculate_take_profit(
        self,
        entry_price: float,
        side: str
    ) -> float:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry price
            side: "long" or "short"

        Returns:
            Take profit price
        """
        if side == "long":
            return entry_price * (1 + self.take_profit_pct / 100)
        else:  # short
            return entry_price * (1 - self.take_profit_pct / 100)

    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        side: str
    ):
        """
        Update trailing stop based on current price.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            entry_price: Position entry price
            side: "long" or "short"
        """
        # Track highest price for longs, lowest for shorts
        if symbol not in self.highest_prices:
            self.highest_prices[symbol] = current_price

        if side == "long":
            # Update highest price
            if current_price > self.highest_prices[symbol]:
                self.highest_prices[symbol] = current_price

            # Calculate trailing stop
            trailing_stop = self.highest_prices[symbol] * (1 - self.trailing_stop_pct / 100)

            # Update if better than current stop
            if symbol not in self.trailing_stops or trailing_stop > self.trailing_stops[symbol]:
                self.trailing_stops[symbol] = trailing_stop

        else:  # short
            # Update lowest price
            if current_price < self.highest_prices[symbol]:
                self.highest_prices[symbol] = current_price

            # Calculate trailing stop
            trailing_stop = self.highest_prices[symbol] * (1 + self.trailing_stop_pct / 100)

            # Update if better than current stop
            if symbol not in self.trailing_stops or trailing_stop < self.trailing_stops[symbol]:
                self.trailing_stops[symbol] = trailing_stop

    def check_stops(
        self,
        symbol: str,
        current_price: float,
        side: str
    ) -> Tuple[bool, str]:
        """
        Check if any stops are triggered.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            side: "long" or "short"

        Returns:
            (triggered, stop_type)
        """
        if side == "long":
            # Check stop loss
            if symbol in self.stop_losses:
                if current_price <= self.stop_losses[symbol]:
                    return True, "stop_loss"

            # Check take profit
            if symbol in self.take_profits:
                if current_price >= self.take_profits[symbol]:
                    return True, "take_profit"

            # Check trailing stop
            if symbol in self.trailing_stops:
                if current_price <= self.trailing_stops[symbol]:
                    return True, "trailing_stop"

        else:  # short
            # Check stop loss
            if symbol in self.stop_losses:
                if current_price >= self.stop_losses[symbol]:
                    return True, "stop_loss"

            # Check take profit
            if symbol in self.take_profits:
                if current_price <= self.take_profits[symbol]:
                    return True, "take_profit"

            # Check trailing stop
            if symbol in self.trailing_stops:
                if current_price >= self.trailing_stops[symbol]:
                    return True, "trailing_stop"

        return False, ""

    def on_position_open(
        self,
        symbol: str,
        entry_price: float,
        side: str
    ):
        """
        Set up stops when position is opened.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            side: "long" or "short"
        """
        # Set stop loss
        self.stop_losses[symbol] = self.calculate_stop_loss(entry_price, side)

        # Set take profit
        self.take_profits[symbol] = self.calculate_take_profit(entry_price, side)

        # Initialize trailing stop tracking
        self.highest_prices[symbol] = entry_price

        logger.info(f"✓ Stops set for {symbol}:")
        logger.info(f"  Entry: {entry_price:.2f}")
        logger.info(f"  Stop loss: {self.stop_losses[symbol]:.2f}")
        logger.info(f"  Take profit: {self.take_profits[symbol]:.2f}")

    def on_position_close(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        side: str,
        size: float,
        stop_type: str = ""
    ):
        """
        Record trade and clean up stops.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            exit_price: Exit price
            side: "long" or "short"
            size: Position size
            stop_type: Type of stop that triggered (if any)
        """
        # Calculate PnL
        if side == "long":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        pnl_usd = (pnl_pct / 100) * (size * entry_price)

        # Record trade
        trade = Trade(
            timestamp=time.time(),
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl_usd,
            pnl_pct=pnl_pct
        )

        self.trade_history.append(trade)

        # Update daily PnL
        self.daily_pnl += pnl_usd

        # Update account balance
        self.account_balance += pnl_usd

        # Update drawdown
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance
        self.current_drawdown_pct = ((self.peak_balance - self.account_balance) / self.peak_balance) * 100

        # Clean up stops
        self.stop_losses.pop(symbol, None)
        self.take_profits.pop(symbol, None)
        self.trailing_stops.pop(symbol, None)
        self.highest_prices.pop(symbol, None)

        # Set cooldown if stopped out
        if stop_type == "stop_loss":
            cooldown_duration = 30 * 60  # 30 minutes
            self.cooldown_until[symbol] = time.time() + cooldown_duration
            logger.warning(f"Stop loss triggered for {symbol}, cooldown for 30 min")

        logger.info(f"✓ Trade closed: {symbol} {side}")
        logger.info(f"  PnL: ${pnl_usd:.2f} ({pnl_pct:.2f}%)")
        logger.info(f"  Account balance: ${self.account_balance:.2f}")

    def reset_daily_stats(self):
        """Reset daily tracking (call at start of each day)."""
        self.daily_pnl = 0.0
        self.daily_start_time = time.time()
        logger.info("Daily stats reset")

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics."""
        if not self.trade_history:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        else:
            wins = [t for t in self.trade_history if t.pnl > 0]
            losses = [t for t in self.trade_history if t.pnl < 0]

            win_rate = len(wins) / len(self.trade_history) * 100

            avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
            avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0.0

            total_wins = sum(t.pnl for t in wins)
            total_losses = sum(abs(t.pnl) for t in losses)
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        return {
            "account_balance": self.account_balance,
            "peak_balance": self.peak_balance,
            "current_drawdown_pct": self.current_drawdown_pct,
            "daily_pnl": self.daily_pnl,
            "total_trades": len(self.trade_history),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor
        }


if __name__ == "__main__":
    # Test risk manager
    risk_mgr = RiskManager(
        max_position_size_usd=10000,
        kelly_fraction=0.25
    )

    # Test Kelly sizing
    size = risk_mgr.calculate_kelly_size(
        confidence=0.7,
        current_price=40000,
        account_balance=10000
    )

    print(f"✓ Kelly position size: {size:.4f} BTC (${size * 40000:.2f})")

    # Test can trade
    can_trade, reason = risk_mgr.check_can_trade("BTC-PERP", 0.7)
    print(f"✓ Can trade: {can_trade} ({reason})")

    # Test stop calculation
    stop_loss = risk_mgr.calculate_stop_loss(40000, "long")
    take_profit = risk_mgr.calculate_take_profit(40000, "long")

    print(f"✓ Stop loss: ${stop_loss:.2f}")
    print(f"✓ Take profit: ${take_profit:.2f}")
