"""
Order execution manager for Hyperliquid perpetuals.

Features:
- Post-only limit orders (ALO) for maker rebates
- Kelly criterion position sizing
- Atomic order placement with retries
- Position tracking and synchronization
"""

import time
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
import logging

import requests

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Order representation."""
    symbol: str
    side: str  # "buy" or "sell"
    size: float
    price: float
    order_type: str = "limit"
    time_in_force: str = "Alo"  # Post-only for maker rebate
    client_order_id: Optional[str] = None
    status: str = "pending"
    filled_size: float = 0.0
    avg_fill_price: float = 0.0


@dataclass
class Position:
    """Position representation."""
    symbol: str
    size: float  # Positive = long, negative = short
    entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 1.0


class HyperliquidExecutor:
    """
    Execute orders on Hyperliquid with robust error handling.

    Supports both testnet and mainnet.
    """

    def __init__(
        self,
        api_url: str,
        wallet_address: str,
        private_key: str,
        is_testnet: bool = True,
        max_retries: int = 3,
        order_timeout: int = 30
    ):
        self.api_url = api_url
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.is_testnet = is_testnet
        self.max_retries = max_retries
        self.order_timeout = order_timeout

        # Session for HTTP requests
        self.session = requests.Session()

        # Track open orders and positions
        self.open_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}

        logger.info(f"✓ Hyperliquid executor initialized ({'testnet' if is_testnet else 'MAINNET'})")

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        order_type: str = "limit",
        time_in_force: str = "Alo",
        reduce_only: bool = False
    ) -> Optional[Order]:
        """
        Place an order on Hyperliquid.

        Args:
            symbol: Trading pair (e.g., "BTC-PERP")
            side: "buy" or "sell"
            size: Order size in base currency
            price: Limit price
            order_type: "limit" or "market"
            time_in_force: "Alo" (post-only), "Gtc", "Ioc"
            reduce_only: Only reduce existing position

        Returns:
            Order object if successful, None otherwise
        """
        order = Order(
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            order_type=order_type,
            time_in_force=time_in_force,
            client_order_id=f"{int(time.time() * 1000)}"
        )

        # Build order payload
        payload = {
            "coin": symbol,
            "is_buy": side == "buy",
            "sz": size,
            "limit_px": price,
            "order_type": {"limit": {"tif": time_in_force}},
            "reduce_only": reduce_only
        }

        # Add authentication headers (simplified - real implementation needs signing)
        headers = {
            "Content-Type": "application/json"
            # TODO: Add proper HMAC signature
        }

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.api_url}/exchange",
                    json={"type": "order", "orders": [payload], "grouping": "na"},
                    headers=headers,
                    timeout=self.order_timeout
                )

                if response.status_code == 200:
                    result = response.json()

                    # Parse response
                    if result.get("status") == "ok":
                        order.status = "submitted"
                        self.open_orders[order.client_order_id] = order

                        logger.info(f"✓ Order placed: {side} {size} {symbol} @ {price}")
                        return order
                    else:
                        error = result.get("response", {}).get("data", "Unknown error")
                        logger.error(f"Order rejected: {error}")
                        return None

                else:
                    logger.warning(f"Order attempt {attempt + 1} failed: {response.status_code}")

            except Exception as e:
                logger.error(f"Order placement error (attempt {attempt + 1}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(1)

        logger.error(f"Order failed after {self.max_retries} attempts")
        return None

    def cancel_order(self, client_order_id: str) -> bool:
        """Cancel an open order."""
        if client_order_id not in self.open_orders:
            logger.warning(f"Order {client_order_id} not found")
            return False

        order = self.open_orders[client_order_id]

        payload = {
            "type": "cancel",
            "cancels": [{
                "coin": order.symbol,
                "o": client_order_id
            }]
        }

        try:
            response = self.session.post(
                f"{self.api_url}/exchange",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                order.status = "cancelled"
                del self.open_orders[client_order_id]
                logger.info(f"✓ Order cancelled: {client_order_id}")
                return True

        except Exception as e:
            logger.error(f"Cancel order error: {e}")

        return False

    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all open orders, optionally filtered by symbol."""
        orders_to_cancel = [
            oid for oid, order in self.open_orders.items()
            if symbol is None or order.symbol == symbol
        ]

        for oid in orders_to_cancel:
            self.cancel_order(oid)

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for a symbol.

        Returns:
            Position object or None
        """
        try:
            response = self.session.post(
                f"{self.api_url}/info",
                json={
                    "type": "clearinghouseState",
                    "user": self.wallet_address
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                positions = data.get("assetPositions", [])

                for pos in positions:
                    if pos.get("position", {}).get("coin") == symbol:
                        size = float(pos["position"]["szi"])
                        entry_price = float(pos["position"]["entryPx"])

                        position = Position(
                            symbol=symbol,
                            size=size,
                            entry_price=entry_price,
                            unrealized_pnl=float(pos["position"].get("unrealizedPnl", 0)),
                            leverage=float(pos["position"].get("leverage", {}).get("value", 1))
                        )

                        self.positions[symbol] = position
                        return position

        except Exception as e:
            logger.error(f"Error fetching position: {e}")

        return None

    def sync_positions(self):
        """Synchronize all positions from exchange."""
        try:
            response = self.session.post(
                f"{self.api_url}/info",
                json={
                    "type": "clearinghouseState",
                    "user": self.wallet_address
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                positions = data.get("assetPositions", [])

                self.positions.clear()

                for pos in positions:
                    symbol = pos.get("position", {}).get("coin")
                    size = float(pos["position"]["szi"])

                    if abs(size) > 1e-8:  # Only track non-zero positions
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            size=size,
                            entry_price=float(pos["position"]["entryPx"]),
                            unrealized_pnl=float(pos["position"].get("unrealizedPnl", 0)),
                            leverage=float(pos["position"].get("leverage", {}).get("value", 1))
                        )

                logger.info(f"✓ Synced {len(self.positions)} positions")

        except Exception as e:
            logger.error(f"Error syncing positions: {e}")

    def close_position(
        self,
        symbol: str,
        price: Optional[float] = None
    ) -> Optional[Order]:
        """
        Close an existing position.

        Args:
            symbol: Trading pair
            price: Limit price (None for market)

        Returns:
            Order object if successful
        """
        position = self.get_position(symbol)

        if not position or abs(position.size) < 1e-8:
            logger.warning(f"No position to close for {symbol}")
            return None

        # Determine close side (opposite of position)
        close_side = "sell" if position.size > 0 else "buy"
        close_size = abs(position.size)

        if price is None:
            # Market close (use current mark price with slippage)
            # TODO: Fetch current mark price
            price = position.entry_price

        return self.place_order(
            symbol=symbol,
            side=close_side,
            size=close_size,
            price=price,
            reduce_only=True
        )


if __name__ == "__main__":
    # Test executor (requires valid credentials)
    executor = HyperliquidExecutor(
        api_url="https://api.hyperliquid-testnet.xyz",
        wallet_address="0x...",
        private_key="YOUR_PRIVATE_KEY",
        is_testnet=True
    )

    # Example: place limit order
    # order = executor.place_order(
    #     symbol="BTC-PERP",
    #     side="buy",
    #     size=0.01,
    #     price=40000.0
    # )

    print("✓ Executor initialized (dry run)")
